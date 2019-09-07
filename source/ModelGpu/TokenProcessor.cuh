#pragma once

#include "sm_60_atomic_functions.h"
#include "device_functions.h"

#include "ModelBasic/ElementaryTypes.h"

#include "Base.cuh"
#include "CudaAccessTOs.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "EnergyGuidance.cuh"
#include "CellComputerFunction.cuh"
#include "PropulsionFunction.cuh"
#include "ScannerFunction.cuh"
#include "ConstructorFunction.cuh"

class TokenProcessor
{
public:
    __inline__ __device__ void init_gridCall(SimulationData& data, int clusterArrayIndex);

    __inline__ __device__ void processingEnergyAveraging_gridCall();
    __inline__ __device__ void processingSpreading_gridCall();
    __inline__ __device__ void processingFeatures_gridCall();

private:
    __inline__ __device__ int calcAnticipatedTokens(Cluster* cluster);
    __inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);
    __inline__ __device__ void moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell);

    __inline__ __device__ void processingCellFeatures(Token* token, EntityFactory& factory);

private:
    SimulationData* _data;
    int _clusterArrayIndex;

    PartitionData _clusterBlock;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void TokenProcessor::init_gridCall(SimulationData& data, int clusterArrayIndex)
{
    _data = &data;
    _clusterArrayIndex = clusterArrayIndex;
    auto const& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    _clusterBlock = calcPartition(clusters.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__inline__ __device__ void TokenProcessor::processingEnergyAveraging_gridCall()
{
    auto const& clusters = _data->entities.clusterPointerArrays.getArray(_clusterArrayIndex);
    for (int clusterIndex = _clusterBlock.startIndex; clusterIndex <= _clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);
        if (0 == cluster->numTokenPointers) {
            continue;
        }

        for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
            Cell* cell = cluster->cellPointers[cellIndex];
            if (1 == cell->alive) {
                cell->tag = 0;
            }
        }

        Cell* candidateCellsForEnergyAveraging[MAX_CELL_BONDS + 1];
        Cell* cellsForEnergyAveraging[MAX_CELL_BONDS + 1];
        for (int tokenIndex = 0; tokenIndex < cluster->numTokenPointers; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            auto& cell = token->cell;
            if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
                continue;
            }

            int tokenBranchNumber = token->memory[0];
            int numCandidateCellsForEnergyAveraging = 1;
            candidateCellsForEnergyAveraging[0] = cell;
            for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
                auto& connectingCell = *cell->connections[connectionIndex];
                if (0 == connectingCell.alive) {
                    continue;
                }
                if (((tokenBranchNumber + 1 - connectingCell.branchNumber)
                    % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                    continue;
                }
                if (connectingCell.tokenBlocked) {
                    continue;
                }
                int numToken = atomicAdd(&connectingCell.tag, 1);
                if (numToken >= cudaSimulationParameters.cellMaxToken) {
                    continue;
                }
                candidateCellsForEnergyAveraging[numCandidateCellsForEnergyAveraging++] = &connectingCell;
            }

            float averageEnergy = 0;
            int numCellsForEnergyAveraging = 0;
            for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                auto const& cell = candidateCellsForEnergyAveraging[index];
                averageEnergy += cell->energy;
                cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
            }
            averageEnergy /= numCellsForEnergyAveraging;
            for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
                auto const& cell = cellsForEnergyAveraging[index];
                cell->energy = averageEnergy;
            }
        }
    }
}

__inline__ __device__ void TokenProcessor::processingSpreading_gridCall()
{
    auto const& clusters = _data->entities.clusterPointerArrays.getArray(_clusterArrayIndex);
    for (int clusterIndex = _clusterBlock.startIndex; clusterIndex <= _clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);
        if (0 == cluster->numTokenPointers) {
            continue;
        }

        int anticipatedTokens = calcAnticipatedTokens(cluster);

        for (int cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
            Cell* cell = cluster->cellPointers[cellIndex];
            if (1 == cell->alive) {
                cell->tag = 0;
            }
        }

        int newNumTokens = 0;
        Token** newTokenPointers = _data->entities.tokenPointers.getNewSubarray(anticipatedTokens);
        for (int tokenIndex = 0; tokenIndex < cluster->numTokenPointers; ++tokenIndex) {
            auto& token = cluster->tokenPointers[tokenIndex];
            auto cell = token->cell;
            if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
                continue;
            }

            int tokenBranchNumber = token->memory[0];

            int numFreePlaces = 0;
            for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
                auto const& connectingCell = cell->connections[connectionIndex];
                if (0 == connectingCell->alive) {
                    continue;
                }
                if (((tokenBranchNumber + 1 - connectingCell->branchNumber)
                    % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                    continue;
                }
                if (connectingCell->tokenBlocked) {
                    continue;
                }
                ++numFreePlaces;
            }

            if (0 == numFreePlaces) {
                cell->energy += token->energy;
                continue;
            }

            float availableTokenEnergyForCell = token->energy / numFreePlaces;
            float remainingTokenEnergy = token->energy;
            bool tokenRecycled = false;
            auto tokenEnergy = token->energy;
            for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
                auto const& connectingCell = cell->connections[connectionIndex];
                if (0 == connectingCell->alive) {
                    continue;
                }
                if (((tokenBranchNumber + 1 - connectingCell->branchNumber)
                    % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                    continue;
                }
                if (connectingCell->tokenBlocked) {
                    continue;
                }
                int numToken = connectingCell->tag++;
                if (numToken >= cudaSimulationParameters.cellMaxToken) {
                    continue;
                }

                int tokenIndex = newNumTokens++;
                Token* newToken;
                if (!tokenRecycled) {
                    moveToken(token, newToken, connectingCell);
                    tokenRecycled = true;
                }
                else {
                    newToken = _data->entities.tokens.getNewElement();
                    copyToken(token, newToken, connectingCell);
                }
                newTokenPointers[tokenIndex] = newToken;

                if (tokenEnergy - availableTokenEnergyForCell > 0) {
                    auto origConnectingCellEnergy = atomicAdd(&connectingCell->energy, -(tokenEnergy - availableTokenEnergyForCell));
                    if (origConnectingCellEnergy > cudaSimulationParameters.cellMinEnergy + tokenEnergy - availableTokenEnergyForCell) {
                        newToken->energy = tokenEnergy;
                    }
                    else {
                        atomicAdd(&connectingCell->energy, tokenEnergy - availableTokenEnergyForCell);
                        newToken->energy = availableTokenEnergyForCell;
                    }
                }
                remainingTokenEnergy -= availableTokenEnergyForCell;
            }
            if (remainingTokenEnergy > 0) {
                cell->energy += remainingTokenEnergy;
            }
        }

        cluster->tokenPointers = newTokenPointers;
        cluster->numTokenPointers = newNumTokens;
    }
}

__inline__ __device__ void TokenProcessor::processingFeatures_gridCall()
{
    EntityFactory factory;
    factory.init(_data);

    auto const& clusters = _data->entities.clusterPointerArrays.getArray(_clusterArrayIndex);
    for (int clusterIndex = _clusterBlock.startIndex; clusterIndex <= _clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        auto const numTokenPointers = cluster->numTokenPointers;
        for (int tokenIndex = 0; tokenIndex < numTokenPointers; ++tokenIndex) {
            auto& token = cluster->tokenPointers[tokenIndex];
            processingCellFeatures(token, factory);
        }
    }
}

__inline__ __device__ int TokenProcessor::calcAnticipatedTokens(Cluster* cluster)
{
    int result = 0;

    for (int tokenIndex = 0; tokenIndex < cluster->numTokenPointers; ++tokenIndex) {
        auto const& token = cluster->tokenPointers[tokenIndex];
        auto& cell = *token->cell;
        if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token->memory[0];
        for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
            auto connectingCell = cell.connections[connectionIndex];
            if (0 == connectingCell->alive) {
                continue;
            }
            if (((tokenBranchNumber + 1 - connectingCell->branchNumber)
                % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                continue;
            }
            if (connectingCell->tokenBlocked) {
                continue;
            }
            ++result;
        }
    }
    return result;
}

__inline__ __device__ void TokenProcessor::copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell)
{
    *targetToken = *sourceToken;
    targetToken->memory[0] = targetCell->branchNumber;
    targetToken->sourceCell = sourceToken->cell;
    targetToken->cell = targetCell;
}

__inline__ __device__ void TokenProcessor::moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell)
{
    targetToken = sourceToken;
    targetToken->memory[0] = targetCell->branchNumber;
    targetToken->sourceCell = sourceToken->cell;
    targetToken->cell = targetCell;
}

__global__ void constructorLauncher(Token* token, SimulationData data)
{
    ConstructorFunction constructor;

    constructor.init_blockCall(token, &data);
    constructor.processing();
}

__inline__ __device__ void TokenProcessor::processingCellFeatures(Token * token, EntityFactory& factory)
{
    auto cell = token->cell;
    EnergyGuidance::processing(token);
    auto type = static_cast<Enums::CellFunction::Type>(cell->cellFunctionType % Enums::CellFunction::_COUNTER);
    switch (type) {
    case Enums::CellFunction::COMPUTER: {
        CellComputerFunction::processing(token);
    } break;
    case Enums::CellFunction::PROPULSION: {
        PropulsionFunction::processing(token, factory);
    } break;
    case Enums::CellFunction::SCANNER: {
        ScannerFunction::processing(token);
    } break;
    case Enums::CellFunction::CONSTRUCTOR: {
        constructorLauncher <<<1, cell->cluster->numCellPointers/*cudaConstants.NUM_THREADS_PER_BLOCK*/>>>(token, *_data);
        cudaDeviceSynchronize();
    } break;
    }
}
