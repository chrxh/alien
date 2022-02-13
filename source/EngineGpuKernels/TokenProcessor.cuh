#pragma once

#include "cuda_runtime_api.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "EnergyGuidance.cuh"
#include "CellComputationProcessor.cuh"
#include "ConstructionProcessor.cuh"
#include "ScannerProcessor.cuh"
#include "DigestionProcessor.cuh"
#include "CommunicationProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "SensorProcessor.cuh"

class TokenProcessor
{
public:
    __inline__ __device__ void movement(SimulationData& data);  //prerequisite: cell tags = 0

    __inline__ __device__ void applyMutation(SimulationData& data);
    __inline__ __device__ void executeReadonlyCellFunctions(SimulationData& data, SimulationResult& result);  //energy values are allowed to change
    __inline__ __device__ void executeModifyingCellFunctions(SimulationData& data, SimulationResult& result);
    __inline__ __device__ void deleteTokenIfCellDeleted(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void TokenProcessor::movement(SimulationData& data)
{
    auto& tokens = data.entities.tokenPointers;
    auto const partition = calcAllThreadsPartition(tokens.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto cell = token->cell;
        atomicAdd(&cell->tokenUsages, 1);

        int numNextTokenCells = 0;
        Cell* nextTokenCells[MAX_CELL_BONDS];
        EntityFactory factory;
        factory.init(&data);

        if (token->energy >= cudaSimulationParameters.tokenMinEnergy) {
            auto tokenBranchNumber = token->getTokenBranchNumber();

            auto cellMinEnergy =
                SpotCalculator::calc(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);

            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (((tokenBranchNumber + 1 - connectedCell->branchNumber) % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                    continue;
                }
                if (connectedCell->tokenBlocked) {
                    continue;
                }

                auto tokenIndex = atomicAdd(&connectedCell->tag, 1);
                if (tokenIndex >= cudaSimulationParameters.cellMaxToken) {
                    continue;
                }
                nextTokenCells[numNextTokenCells++] = connectedCell;
            }

            token->energy = numNextTokenCells > 0 ? token->energy / numNextTokenCells : token->energy;

            for (int i = 0; i < numNextTokenCells; ++i) {
                auto const& connectedCell = nextTokenCells[i];

                SystemDoubleLock lock;
                lock.init(&cell->locked, &connectedCell->locked);
                if (lock.tryLock()) {
                    auto averageEnergy = (cell->energy + connectedCell->energy) / 2;
                    cell->energy = averageEnergy;
                    connectedCell->energy = averageEnergy;
                    lock.releaseLock();
                }

                token->memory[Enums::Branching_TokenBranchNumber] = connectedCell->branchNumber;
                if (0 == i) {
                    token->sourceCell = token->cell;
                    token->cell = connectedCell;
                } else {
                    factory.duplicateToken(connectedCell, token);
                }
            }
        }
        if (0 == numNextTokenCells) {
            if (cell->tryLock()) {
                cell->energy += token->energy;
                token = nullptr;
                cell->releaseLock();
            }
        }
    }
}

__inline__ __device__ void TokenProcessor::applyMutation(SimulationData& data)
{
    auto& tokens = data.entities.tokenPointers;
    auto const partition = calcAllThreadsPartition(tokens.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto const& cell = token->cell;
        auto mutationRate = SpotCalculator::calc(&SimulationParametersSpotValues::tokenMutationRate, data, cell->absPos);
        if (data.numberGen1.random() < mutationRate) {
            token->memory[data.numberGen1.random(MAX_TOKEN_MEM_SIZE - 1)] = data.numberGen1.random(255);
        }
    }
}

__inline__ __device__ void TokenProcessor::executeReadonlyCellFunctions(SimulationData& data, SimulationResult& result)
{
    auto& tokens = data.entities.tokenPointers;
    auto partition =
        calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto& cell = token->cell;
        if (token) {
            auto cellFunctionType = cell->getCellFunctionType();
            if (cell->tryLock()) {
                if (Enums::CellFunction_Scanner == cellFunctionType) {
                    ScannerProcessor::process(token, data);
                }
                if (Enums::CellFunction_Digestion == cellFunctionType) {
                    DigestionProcessor::process(token, data, result);
                }
                if (Enums::CellFunction_Sensor == cellFunctionType) {
                    SensorProcessor::scheduleOperation(token, data);
                }
                cell->releaseLock();
            }
        }
    }
}

__inline__ __device__ void
TokenProcessor::executeModifyingCellFunctions(SimulationData& data, SimulationResult& result)
{
    auto& tokens = data.entities.tokenPointers;
    auto const partition = calcAllThreadsPartition(tokens.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto& cell = token->cell;
        if (token) {
            auto cellFunctionType = cell->getCellFunctionType();
            //IMPORTANT:
            //loop would lead to time out problems on GeForce 10-series
/*
            bool success = false;
            do {
*/
                if (cell->tryLock()) {

                    EnergyGuidance::processing(data, token);
                    if (Enums::CellFunction_Computation== cellFunctionType) {
                        CellComputationProcessor::process(token);
                    }
                    if (Enums::CellFunction_Constructor == cellFunctionType) {
                        ConstructionProcessor::process(token, data, result);
                    }
                    if (Enums::CellFunction_Communication == cellFunctionType) {
                        CommunicationProcessor::process(token, data);
                    }
                    if (Enums::CellFunction_Muscle == cellFunctionType) {
                        MuscleProcessor::process(token, data, result);
                    }

                    //                    success = true;
                    cell->releaseLock();
                }
/*            } while (!success);*/
        }
    }
}

__inline__ __device__ void TokenProcessor::deleteTokenIfCellDeleted(SimulationData& data)
{
    auto& tokens = data.entities.tokenPointers;
    auto partition =
        calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (auto& token = tokens.at(index)) {
            auto& cell = token->cell;
            if (cell->isDeleted()) {
                EntityFactory factory;
                factory.init(&data);
                factory.createParticle(token->energy, cell->absPos, cell->vel, {cell->metadata.color});

                token = nullptr;
            }
        }
    }
}