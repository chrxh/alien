#pragma once

#include "cuda_runtime_api.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "EnergyGuidance.cuh"
#include "ComputationFunction.cuh"
#include "ConstructorFunction.cuh"
#include "ScannerFunction.cuh"
#include "DigestionFunction.cuh"
#include "PropulsionFunction.cuh"
#include "MuscleFunction.cuh"

class TokenProcessor
{
public:
    __inline__ __device__ void movement(SimulationData& data);  //prerequisite: clearTag, need numTokenPointers because it might be changed

    __inline__ __device__ void executeReadonlyCellFunctions(SimulationData& data, SimulationResult& result);
    __inline__ __device__ void
    executeModifyingCellFunctions(SimulationData& data, SimulationResult& result, int numTokenPointers);
    __inline__ __device__ void deleteTokenIfCellDeleted(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void TokenProcessor::movement(SimulationData& data)
{
    auto& tokens = data.entities.tokenPointers;
    auto partition = calcPartition(data.originalArraySizes->tokenArraySize, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto cell = token->cell;
        atomicAdd(&cell->tokenUsages, 1);

        int numMovedTokens = 0;
        EntityFactory factory;
        factory.init(&data);

        if (token->energy >= cudaSimulationParameters.tokenMinEnergy) {
            auto tokenBranchNumber = token->getTokenBranchNumber();

            auto cellMinEnergy =
                SpotCalculator::calc(&SimulationParametersSpotValues::cellMinEnergy, data, cell->absPos);
            auto tokenMutationRate =
                SpotCalculator::calc(&SimulationParametersSpotValues::tokenMutationRate, data, cell->absPos);

            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (((tokenBranchNumber + 1 - connectedCell->branchNumber)
                     % cudaSimulationParameters.cellMaxTokenBranchNumber)
                    != 0) {
                    continue;
                }
                if (connectedCell->tokenBlocked) {
                    continue;
                }

                auto tokenIndex = atomicAdd(&connectedCell->tag, 1);
                if (tokenIndex >= cudaSimulationParameters.cellMaxToken) {
                    continue;
                }

                token->memory[Enums::Branching::TOKEN_BRANCH_NUMBER] = connectedCell->branchNumber;
                if (0 == numMovedTokens) {
                    token->sourceCell = token->cell;
                    token->cell = connectedCell;
                    ++numMovedTokens;

                    
                    if (data.numberGen.random() < tokenMutationRate) {
                        token->memory[data.numberGen.random(MAX_TOKEN_MEM_SIZE - 1)] = data.numberGen.random(255);
                    }
                } else {
                    auto origEnergy = atomicAdd(&connectedCell->energy, -token->energy); 
                    if (origEnergy > cellMinEnergy + token->energy) {
                        factory.duplicateToken(connectedCell, token);
                        ++numMovedTokens;
                    } else {
                        atomicAdd(&connectedCell->energy, token->energy); 
                    }
                }
            }
        }
        if (0 == numMovedTokens) {
            atomicAdd(&cell->energy, token->energy);
            token = nullptr;
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
                if (Enums::CellFunction::SCANNER == cellFunctionType) {
                    ScannerFunction::processing(token, data);
                }
                if (Enums::CellFunction::DIGESTION == cellFunctionType) {
                    DigestionFunction::processing(token, data, result);
                }
                cell->releaseLock();
            }
        }
    }
}

__inline__ __device__ void
TokenProcessor::executeModifyingCellFunctions(SimulationData& data, SimulationResult& result, int numTokenPointers)
{
    auto& tokens = data.entities.tokenPointers;
    auto partition = calcPartition(numTokenPointers, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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
                    if (Enums::CellFunction::COMPUTATION == cellFunctionType) {
                        ComputationFunction::processing(token);
                    }
                    if (Enums::CellFunction::CONSTRUCTOR == cellFunctionType) {
                        ConstructorFunction::processing(token, data, result);
                    }
                    if (Enums::CellFunction::PROPULSION == cellFunctionType) {
                        PropulsionFunction::processing(token, data);
                    }
                    if (Enums::CellFunction::MUSCLE == cellFunctionType) {
                        MuscleFunction::processing(token, data, result);
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