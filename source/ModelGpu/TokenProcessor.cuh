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

class TokenProcessor
{
public:
    __inline__ __device__ void init_blockCall(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingEnergyAveraging_blockCall();
    __inline__ __device__ void processingSpreading_blockCall();
    __inline__ __device__ void processingFeatures_blockCall();

private:
    __inline__ __device__ void calcAnticipatedTokens_blockCall(int& result);
    __inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);
    __inline__ __device__ void moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell);

    __inline__ __device__ void processingCellFeatures(Token* token, EntityFactory& factory);

private:
    SimulationData* _data;
    Cluster* _cluster;

    int _startCellIndex;
    int _endCellIndex;
    int _startTokenIndex;
    int _endTokenIndex;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void TokenProcessor::init_blockCall(SimulationData& data, int clusterIndex)
{
    _data = &data;
    _cluster = &data.clusters.getEntireArray()[clusterIndex];

    calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x, _startCellIndex, _endCellIndex);
    calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x, _startTokenIndex, _endTokenIndex);
}

__inline__ __device__ void TokenProcessor::processingEnergyAveraging_blockCall()
{
    if (0 == _cluster->numTokenPointers) {
        return;
    }

    for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
        Cell* cell = _cluster->cellPointers[cellIndex];
        if (cell->alive) {
            cell->tag = 0;
        }
    }
    __syncthreads();

    Cell* candidateCellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    Cell* cellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        auto const& token = _cluster->tokenPointers[tokenIndex];
        auto& cell = token->cell;
        if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token->memory[0];
        int numCandidateCellsForEnergyAveraging = 1;
        candidateCellsForEnergyAveraging[0] = cell;
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto& connectingCell = *cell->connections[connectionIndex];
            if (!connectingCell.alive) {
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
            int locked = atomicExch(&cell->locked, 1);
            if (0 == locked) {
                averageEnergy += cell->energy;
                cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
            }
        }
        averageEnergy /= numCellsForEnergyAveraging;
        for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
            auto const& cell = cellsForEnergyAveraging[index];
            cell->energy = averageEnergy;
            atomicExch(&cell->locked, 0);
        }
    }
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::processingSpreading_blockCall()
{
    if (0 == _cluster->numTokenPointers) {
        return;
    }

    __shared__ int anticipatedTokens;
    calcAnticipatedTokens_blockCall(anticipatedTokens);

    __shared__ Token** newTokenPointers;
    __shared__ int newNumTokens;
    if (0 == threadIdx.x) {
        newNumTokens = 0;
        newTokenPointers = _data->tokenPointers.getNewSubarray(anticipatedTokens);
    }
    for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
        Cell* cell = _cluster->cellPointers[cellIndex];
        if (cell->alive) {
            cell->tag = 0;
        }
    }
    __syncthreads();

    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        auto& token = _cluster->tokenPointers[tokenIndex];
        auto cell = token->cell;
        if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token->memory[0];

        int numFreePlaces = 0;
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            if (!connectingCell->alive) {
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
            atomicAdd(&cell->energy, token->energy);
            continue;
        }

        float availableTokenEnergyForCell = token->energy / numFreePlaces;
        float remainingTokenEnergy = token->energy;
        bool tokenRecycled = false;
        auto tokenEnergy = token->energy;
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            if (!connectingCell->alive) {
                continue;
            }
            if (((tokenBranchNumber + 1 - connectingCell->branchNumber)
                % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                continue;
            }
            if (connectingCell->tokenBlocked) {
                continue;
            }
            int numToken = atomicAdd(&connectingCell->tag, 1);
            if (numToken >= cudaSimulationParameters.cellMaxToken) {
                continue;
            }

            int tokenIndex = atomicAdd(&newNumTokens, 1);
            Token* newToken;
            if (!tokenRecycled) {
                moveToken(token, newToken, connectingCell);
                tokenRecycled = true;
            }
            else {
                newToken = _data->tokens.getNewElement();
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
            atomicAdd(&cell->energy, remainingTokenEnergy);
        }
    }
    __syncthreads();


    if (0 == threadIdx.x) {
        _cluster->tokenPointers = newTokenPointers;
        _cluster->numTokenPointers = newNumTokens;
    }
    __syncthreads();

    calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x, _startTokenIndex, _endTokenIndex);
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::processingFeatures_blockCall()
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(_data);
    }
    __syncthreads();

    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        auto& token = _cluster->tokenPointers[tokenIndex];
        processingCellFeatures(token, factory);
    }
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::calcAnticipatedTokens_blockCall(int& result)
{
    if (0 == threadIdx.x) {
        result = 0;
    }
    __syncthreads();

    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        auto const& token = _cluster->tokenPointers[tokenIndex];
        auto& cell = *token->cell;
        if (token->energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token->memory[0];
        for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
            auto connectingCell = cell.connections[connectionIndex];
            if (!connectingCell->alive) {
                continue;
            }
            if (((tokenBranchNumber + 1 - connectingCell->branchNumber)
                % cudaSimulationParameters.cellMaxTokenBranchNumber) != 0) {
                continue;
            }
            if (connectingCell->tokenBlocked) {
                continue;
            }
            atomicAdd(&result, 1);
        }
    }
    __syncthreads();
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

__inline__ __device__ void TokenProcessor::processingCellFeatures(Token * token, EntityFactory& factory)
{
    auto cell = token->cell;
    int locked;
    do {    //mutex
        locked = atomicExch(&cell->locked, 1);
        if (0 == locked) {
            EnergyGuidance::processing(token);
            auto type = static_cast<Enums::CellFunction::Type>(cell->cellFunctionType % Enums::CellFunction::_COUNTER);
            switch (type) {
            case Enums::CellFunction::COMPUTER: {
                CellComputerFunction::processing(token);
            } break;
            case Enums::CellFunction::PROPULSION: {
                PropulsionFunction::processing(token, factory);
            } break;
            }
            atomicExch(&cell->locked, 0);
        }
    } while (1 == locked);

}

