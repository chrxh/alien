#pragma once

#include "sm_60_atomic_functions.h"
#include "device_functions.h"

#include "ModelBasic/ElementaryTypes.h"

#include "Base.cuh"
#include "AccessTOs.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "EnergyGuidance.cuh"
#include "CellComputerFunction.cuh"
#include "PropulsionFunction.cuh"
#include "ScannerFunction.cuh"
#include "ConstructorFunction.cuh"
#include "WeaponFunction.cuh"
#include "SensorFunction.cuh"
#include "CommunicatorFunction.cuh"

class TokenProcessor
{
public:
    __inline__ __device__ void init_gridCall(SimulationData& data);
    __inline__ __device__ void init_blockCall(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingEnergyAveraging_gridCall();
    __inline__ __device__ void processingSpreading_gridCall();
    __inline__ __device__ void processingLightWeigthedFeatures_gridCall();
    __inline__ __device__ void processingHeavyWeightedFeatures_blockCall();

    __inline__ __device__ void createCellFunctionData_blockCall();

private:
    __inline__ __device__ void calcAnticipatedTokens(Cluster* cluster, int& result);
    __inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);
    __inline__ __device__ void moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell, Cell* dummy);

private:
    SimulationData* _data;
    Cluster* _cluster;
    PartitionData _clusterPartition;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void TokenProcessor::init_gridCall(SimulationData& data)
{
    _data = &data;
    auto const& clusters = data.entities.clusterPointers;
    _clusterPartition =
        calcPartition(clusters.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__inline__ __device__ void TokenProcessor::init_blockCall(SimulationData& data, int clusterIndex)
{
    _data = &data;
    _cluster = data.entities.clusterPointers.at(clusterIndex);
}

__inline__ __device__ void TokenProcessor::processingEnergyAveraging_gridCall()
{
//    auto const& clusters = _data->entities.clusterPointers;
//    for (int clusterIndex = _clusterPartition.startIndex; clusterIndex <= _clusterPartition.endIndex; ++clusterIndex) {
        auto& cluster = _cluster;
        if (0 == cluster->numTokenPointers) {
            return;
        }
        if (0 == threadIdx.x) {

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
                if (token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
                    continue;
                }

                int tokenBranchNumber = token->getTokenBranchNumber();
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
                    cell->getLock();
                }

                for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                    auto const& cell = candidateCellsForEnergyAveraging[index];
                    averageEnergy += cell->getEnergy();
                    cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
                }
                averageEnergy /= numCellsForEnergyAveraging;
                for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
                    auto const& cell = cellsForEnergyAveraging[index];
                    cell->setEnergy(averageEnergy);
                }

                for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                    auto const& cell = candidateCellsForEnergyAveraging[index];
                    cell->releaseLock();
                }
            }
        }
        __syncthreads();
//    }
}

__inline__ __device__ void TokenProcessor::processingSpreading_gridCall()
{
    auto& cluster = _cluster;
    if (0 == cluster->numTokenPointers) {
        return;
    }

    __shared__ int anticipatedTokens;
    calcAnticipatedTokens(cluster, anticipatedTokens);

    __shared__ int clusterLock;
    if (0 == threadIdx.x) {
        clusterLock = 0;
    }
    __syncthreads();

    auto const cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
    for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        auto& cell = cluster->cellPointers[cellIndex];
        if (1 == cell->alive) {
            cell->tag = 0;
        }
    }
    __syncthreads();

    __shared__ int newNumTokens;
    __shared__ Token** newTokenPointers;
    if (0 == threadIdx.x) {
        newNumTokens = 0;
        newTokenPointers = _data->entities.tokenPointers.getNewSubarray(anticipatedTokens);
    }
    __syncthreads();

    auto const tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
    for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto& token = cluster->tokenPointers[tokenIndex];
        auto cell = token->cell;
        if (0 == cell->alive || token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
            cell->getLock();
            cell->changeEnergy(token->getEnergy());
            cell->releaseLock();
            continue;
        }

        auto const tokenBranchNumber = token->getTokenBranchNumber();

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
            cell->getLock();
            cell->changeEnergy(token->getEnergy());
            cell->releaseLock();
            continue;
        }

        while (1 == atomicExch(&clusterLock, 1)) {}
        cell->getLock();
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            connectingCell->getLock();
        }
        atomicExch(&clusterLock, 0);

        auto const tokenEnergy = token->getEnergy();
        auto const sharedEnergyFromPrevToken = tokenEnergy / numFreePlaces;
        auto remainingTokenEnergyForCell = tokenEnergy;
        auto tokenRecycled = false;
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
            ++connectingCell->tokenUsages;
            if (!tokenRecycled) {
                moveToken(token, newToken, connectingCell, cell);
                tokenRecycled = true;
            }
            else {
                newToken = _data->entities.tokens.getNewElement();
                copyToken(token, newToken, connectingCell);
            }
            newTokenPointers[tokenIndex] = newToken;

            if (connectingCell->getEnergy() > cudaSimulationParameters.cellMinEnergy + tokenEnergy - sharedEnergyFromPrevToken) {
                newToken->setEnergy(tokenEnergy);
                connectingCell->changeEnergy(-(tokenEnergy - sharedEnergyFromPrevToken));
            }
            else {
                newToken->setEnergy(sharedEnergyFromPrevToken);
            }
            remainingTokenEnergyForCell -= sharedEnergyFromPrevToken;
        }
        if (remainingTokenEnergyForCell > 0) {
            cell->changeEnergy(remainingTokenEnergyForCell);
        }
        cell->releaseLock();
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            connectingCell->releaseLock();
        }
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        cluster->tokenPointers = newTokenPointers;
        cluster->numTokenPointers = newNumTokens;
    }
    __syncthreads();



/*
        if(0 == threadIdx.x) {
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
                if (0 == cell->alive || token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
                    cell->getLock();
                    cell->changeEnergy(token->getEnergy());
                    cell->releaseLock();
                    continue;
                }

                int tokenBranchNumber = token->getTokenBranchNumber();

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
                    cell->getLock();
                    cell->changeEnergy(token->getEnergy());
                    cell->releaseLock();
                    continue;
                }

                cell->getLock();
                for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
                    auto const& connectingCell = cell->connections[connectionIndex];
                    connectingCell->getLock();
                }

                auto const tokenEnergy = token->getEnergy();
                auto const sharedEnergyFromPrevToken = tokenEnergy / numFreePlaces;
                auto remainingTokenEnergyForCell = tokenEnergy;
                auto tokenRecycled = false;
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
                    ++connectingCell->tokenUsages;
                    if (!tokenRecycled) {
                        moveToken(token, newToken, connectingCell);
                        tokenRecycled = true;
                    }
                    else {
                        newToken = _data->entities.tokens.getNewElement();
                        copyToken(token, newToken, connectingCell);
                    }
                    newTokenPointers[tokenIndex] = newToken;

                    if (connectingCell->getEnergy() > cudaSimulationParameters.cellMinEnergy + tokenEnergy - sharedEnergyFromPrevToken) {
                        newToken->setEnergy(tokenEnergy);
                        connectingCell->changeEnergy(-(tokenEnergy - sharedEnergyFromPrevToken));
                    }
                    else {
                        newToken->setEnergy(sharedEnergyFromPrevToken);
                    }
                    remainingTokenEnergyForCell -= sharedEnergyFromPrevToken;
                }
                if (remainingTokenEnergyForCell > 0) {
                    cell->changeEnergy(remainingTokenEnergyForCell);
                }
                cell->releaseLock();
                for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
                    auto const& connectingCell = cell->connections[connectionIndex];
                    connectingCell->releaseLock();
                }
            }

            cluster->tokenPointers = newTokenPointers;
            cluster->numTokenPointers = newNumTokens;
        }
        __syncthreads();
*/


}

__inline__ __device__ void TokenProcessor::processingLightWeigthedFeatures_gridCall()
{
    EntityFactory factory;
    factory.init(_data);

    if(0 == threadIdx.x) {
        auto& cluster = _cluster;
        auto const numTokenPointers = cluster->numTokenPointers;
        for (int tokenIndex = 0; tokenIndex < numTokenPointers; ++tokenIndex) {
            auto& token = cluster->tokenPointers[tokenIndex];
            auto cell = token->cell;
            cell->getLock();
            EnergyGuidance::processing(token);
            switch (cell->getCellFunctionType()) {
            case Enums::CellFunction::COMPUTER: {
                CellComputerFunction::processing(token);
            } break;
            case Enums::CellFunction::PROPULSION: {
                PropulsionFunction::processing(token, factory);
            } break;
            case Enums::CellFunction::SCANNER: {
                ScannerFunction::processing(token);
            } break;
            case Enums::CellFunction::WEAPON: {
                WeaponFunction::processing(token, _data);
            } break;
            }
            cell->releaseLock();
        }
    }
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::processingHeavyWeightedFeatures_blockCall()
{
    __syncthreads();

    auto const numTokenPointers = _cluster->numTokenPointers;
    if (0 == numTokenPointers) {
        return;
    }
    ConstructorFunction constructor;
    constructor.init_blockCall(_cluster, _data);

    SensorFunction sensor;
    sensor.init_blockCall(_cluster, _data);

    CommunicatorFunction communicator;
    communicator.init_blockCall(_data);
    __syncthreads();

    for (int tokenIndex = 0; tokenIndex < numTokenPointers; ++tokenIndex) {
        auto const& token = _cluster->tokenPointers[tokenIndex];

        auto const type = token->cell->getCellFunctionType();
        switch (type) {
        case Enums::CellFunction::CONSTRUCTOR: {
            __syncthreads();

            constructor.processing_blockCall(token);
            __syncthreads();
        } break;
        case Enums::CellFunction::SENSOR: {
            sensor.processing_blockCall(token);
            __syncthreads();
        } break;
        case Enums::CellFunction::COMMUNICATOR: {
            communicator.processing_blockCall(token);
            __syncthreads();
        } break;
        }
    }
}

__inline__ __device__ void TokenProcessor::createCellFunctionData_blockCall()
{
    auto const cellPartition = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
    __shared__ bool hasToken;
    __shared__ bool hasCommunicator;
    if (0 == threadIdx.x) {
        hasToken = _cluster->numTokenPointers > 0;
        hasCommunicator = false;
    }
    __syncthreads();

    for (int cellIndex = cellPartition.startIndex; cellIndex <= cellPartition.endIndex; ++cellIndex) {
        auto const& cell = _cluster->cellPointers[cellIndex];
        if (Enums::CellFunction::COMMUNICATOR == cell->getCellFunctionType()) {
            hasCommunicator = true;
        }
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        if (hasToken && hasCommunicator) {
            _data->cellFunctionData.mapSectionCollector.insert(_cluster, &_data->dynamicMemory);
        }
    }
}

__inline__ __device__ void TokenProcessor::calcAnticipatedTokens(Cluster* cluster, int& result)
{
    if (0 == threadIdx.x) {
        result = 0;
    }
    __syncthreads();

    auto const tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
    for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto const& token = cluster->tokenPointers[tokenIndex];
        auto const& cell = *token->cell;
        if (token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        auto const tokenBranchNumber = token->getTokenBranchNumber();
        for (auto connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
            auto const& connectingCell = cell.connections[connectionIndex];
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
            atomicAdd_block(&result, 1);
        }
    }
    __syncthreads();
/*
    int result = 0;
    for (int tokenIndex = 0; tokenIndex < cluster->numTokenPointers; ++tokenIndex) {
        auto const& token = cluster->tokenPointers[tokenIndex];
        auto& cell = *token->cell;
        if (token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token->getTokenBranchNumber();
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
*/
}

__inline__ __device__ void TokenProcessor::copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell)
{
    *targetToken = *sourceToken;
    targetToken->memory[0] = targetCell->branchNumber;
    targetToken->sourceCell = sourceToken->cell;
    targetToken->cell = targetCell;
}

__inline__ __device__ void TokenProcessor::moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell, Cell* dummy)
{
    targetToken = sourceToken;
    targetToken->memory[0] = targetCell->branchNumber;
    targetToken->sourceCell = sourceToken->cell;
    targetToken->cell = targetCell;
}
