#pragma once

#include "cuda_runtime_api.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "EnergyGuidance.cuh"
#include "CellComputerFunction.cuh"
#include "ConstructorFunction.cuh"

class TokenProcessor
{
public:
    __inline__ __device__ void movement(
        SimulationData& data,
        int numTokenPointers);  //prerequisite: clearTag, need numTokenPointers because it might be changed

    __inline__ __device__ void executeCellFunctions(SimulationData& data, int numTokenPointers);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void TokenProcessor::movement(SimulationData& data, int numTokenPointers)
{
    auto& tokens = data.entities.tokenPointers;
    auto partition = calcPartition(numTokenPointers, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    Cell* cellsForEnergyAveraging[MAX_CELL_BONDS + 1];

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& token = tokens.at(index);
        auto cell = token->cell;
        int numMovedTokens = 0;
        EntityFactory factory;
        factory.init(&data);
        int numCellForEnergyAveraging = 0;
        if (cell->tryLock()) {
            cellsForEnergyAveraging[numCellForEnergyAveraging++] = cell;
        }
        auto tokenBranchNumber = token->getTokenBranchNumber();
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
            } else {
                factory.createToken(connectedCell, token);
            }

            if (connectedCell->tryLock()) {
                cellsForEnergyAveraging[numCellForEnergyAveraging++] = connectedCell;
            }
            ++numMovedTokens;
        }
        if (0 == numMovedTokens) {
            token = nullptr;
        }
        if (numCellForEnergyAveraging > 0) {
            float averageEnergy = 0;
            for (int i = 0; i < numCellForEnergyAveraging; ++i) {
                averageEnergy += cellsForEnergyAveraging[i]->energy;
            }
            averageEnergy /= numCellForEnergyAveraging;
            for (int i = 0; i < numCellForEnergyAveraging; ++i) {
                cellsForEnergyAveraging[i]->energy = averageEnergy;
                cellsForEnergyAveraging[i]->releaseLock();
            }
        }
    }
}

__inline__ __device__ void
TokenProcessor::executeCellFunctions(SimulationData& data, int numTokenPointers)
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

                    EnergyGuidance::processing(token);
                    if (Enums::CellFunction::COMPUTER == cellFunctionType) {
                        CellComputerFunction::processing(token);
                    }
                    if (Enums::CellFunction::CONSTRUCTOR == cellFunctionType) {
                        ConstructorFunction::processing(token, data);
                    }

//                    success = true;
                    cell->releaseLock();
                }
/*            } while (!success);*/
        }
    }
}


/*
#pragma once

#include "sm_60_atomic_functions.h"
#include "cuda_runtime_api.h"

#include "EngineInterface/ElementaryTypes.h"

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
    __inline__ __device__ void init_block(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingEnergyAveraging_block();
    __inline__ __device__ void processingSpreading_block();
    __inline__ __device__ void processingLightWeigthedFeatures_block();

    __inline__ __device__ void createCellFunctionData_block();

    __inline__ __device__ void processingConstructors_block();

    __inline__ __device__ void processingCommunicatorsAnsSensors_block();

    __inline__ __device__ void repair_block();

private:
    __inline__ __device__ void calcAnticipatedTokens(Cluster* cluster, int& result);
    __inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);
    __inline__ __device__ void moveToken(Token* sourceToken, Token*& targetToken, Cell* targetCell, Cell* dummy);

    __inline__ __device__ void resetTags_block();

private:
    SimulationData* _data;
    Cluster* _cluster;
    PartitionData _cellPartition;
    PartitionData _tokenPartition;
};

/ ************************************************************************ /
/ * Implementation                                                       * /
/ ************************************************************************ /
__inline__ __device__ void TokenProcessor::init_block(SimulationData& data, int clusterIndex)
{
    _data = &data;
    _cluster = data.entities.clusterPointers.at(clusterIndex);
    _cellPartition = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
    _tokenPartition = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
}

__inline__ __device__ void TokenProcessor::processingEnergyAveraging_block()
{
    auto& cluster = _cluster;
    if (0 == cluster->numTokenPointers) {
        return;
    }

    __shared__ BlockLock clusterLock;
    clusterLock.init_block();

    resetTags_block();

    Cell* candidateCellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    Cell* cellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    for (auto tokenIndex = _tokenPartition.startIndex; tokenIndex <= _tokenPartition.endIndex; ++tokenIndex) {
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

        bool clusterLockResult = false;
        do {
            if (clusterLockResult = clusterLock.tryLock()) {
                for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                    auto const& cell = candidateCellsForEnergyAveraging[index];
                    cell->getLock();
                }
                clusterLock.releaseLock();

                for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                    auto const& cell = candidateCellsForEnergyAveraging[index];
                    averageEnergy += cell->getEnergy_safe();
                    cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
                }
                averageEnergy /= numCellsForEnergyAveraging;
                for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
                    auto const& cell = cellsForEnergyAveraging[index];
                    cell->setEnergy_safe(averageEnergy);
                }

                for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
                    auto const& cell = candidateCellsForEnergyAveraging[index];
                    cell->releaseLock();
                }
            }
        } while (!clusterLockResult);
    }
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::processingSpreading_block()
{
    if (0 == _cluster->numTokenPointers) {
        return;
    }

    __shared__ int anticipatedTokens;
    calcAnticipatedTokens(_cluster, anticipatedTokens);

    __shared__ BlockLock clusterLock;
    clusterLock.init_block();

    resetTags_block();

    __shared__ int newNumTokens;
    __shared__ Token** newTokenPointers;
    if (0 == threadIdx.x) {
        newNumTokens = 0;
        newTokenPointers = _data->entities.tokenPointers.getNewSubarray(anticipatedTokens);
    }
    __syncthreads();

    for (auto tokenIndex = _tokenPartition.startIndex; tokenIndex <= _tokenPartition.endIndex; ++tokenIndex) {
        auto& token = _cluster->tokenPointers[tokenIndex];

        auto cell = token->cell;
        if (0 == cell->alive || token->getEnergy() < cudaSimulationParameters.tokenMinEnergy) {
            cell->getLock();
            cell->changeEnergy_safe(token->getEnergy());
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
            cell->changeEnergy_safe(token->getEnergy());
            cell->releaseLock();
            continue;
        }

        clusterLock.getLock();
        cell->getLock();
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            connectingCell->getLock();
        }
        clusterLock.releaseLock();

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
            
            int numToken = atomicAdd_block(&connectingCell->tag, 1);
            if (numToken >= cudaSimulationParameters.cellMaxToken) {
                continue;
            }

            int tokenIndex = atomicAdd_block(&newNumTokens, 1);
            Token* newToken;
            atomicAdd_block(&connectingCell->tokenUsages, 1);
            if (!tokenRecycled) {
                moveToken(token, newToken, connectingCell, cell);
                tokenRecycled = true;
            }
            else {
                newToken = _data->entities.tokens.getNewElement();
                copyToken(token, newToken, connectingCell);
            }
            newTokenPointers[tokenIndex] = newToken;

            if (connectingCell->getEnergy_safe() > cudaSimulationParameters.cellMinEnergy + tokenEnergy - sharedEnergyFromPrevToken) {
                newToken->setEnergy(tokenEnergy);
                connectingCell->changeEnergy_safe(-(tokenEnergy - sharedEnergyFromPrevToken));
            }
            else {
                newToken->setEnergy(sharedEnergyFromPrevToken);
            }
            remainingTokenEnergyForCell -= sharedEnergyFromPrevToken;
        }
        if (remainingTokenEnergyForCell > 0) {
            cell->changeEnergy_safe(remainingTokenEnergyForCell);
        }

        cell->releaseLock();
        for (int connectionIndex = 0; connectionIndex < cell->numConnections; ++connectionIndex) {
            auto const& connectingCell = cell->connections[connectionIndex];
            connectingCell->releaseLock();
        }
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        _cluster->tokenPointers = newTokenPointers;
        _cluster->numTokenPointers = newNumTokens;
    }
    __syncthreads();

    _tokenPartition = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
}

__inline__ __device__ void TokenProcessor::processingLightWeigthedFeatures_block()
{
    EntityFactory factory;
    factory.init(_data);

    for (auto tokenIndex = _tokenPartition.startIndex; tokenIndex <= _tokenPartition.endIndex; ++tokenIndex) {
        auto& token = _cluster->tokenPointers[tokenIndex];
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
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::processingConstructors_block()
{
    __syncthreads();

    auto const numTokenPointers = _cluster->numTokenPointers;
    if (0 == numTokenPointers) {
        __syncthreads();
        return;
    }

    ConstructorFunction constructor;
    constructor.init_block(_cluster, _data);

    __syncthreads();

    for (int tokenIndex = 0; tokenIndex < numTokenPointers; ++tokenIndex) {
        auto const& token = _cluster->tokenPointers[tokenIndex];
        auto const type = token->cell->getCellFunctionType();
        __syncthreads();
        switch (type) {
        case Enums::CellFunction::CONSTRUCTOR: {
            constructor.processing_block(token);
        } break;
        }
        __syncthreads();
    }
}

__inline__ __device__ void TokenProcessor::processingCommunicatorsAnsSensors_block()
{
    __syncthreads();

    auto const numTokenPointers = _cluster->numTokenPointers;
    if (0 == numTokenPointers) {
        __syncthreads();
        return;
    }

    SensorFunction sensor;
    sensor.init_block(_cluster, _data);

    CommunicatorFunction communicator;
    communicator.init_block(_data);
    __syncthreads();

    for (int tokenIndex = 0; tokenIndex < numTokenPointers; ++tokenIndex) {
        auto const& token = _cluster->tokenPointers[tokenIndex];
        auto const type = token->cell->getCellFunctionType();
        __syncthreads();
        switch (type) {
        case Enums::CellFunction::SENSOR: {
            sensor.processing_block(token);
        } break;
        case Enums::CellFunction::COMMUNICATOR: {
            communicator.processing_block(token);
        } break;
        }
        __syncthreads();
    }
}

__inline__ __device__ void TokenProcessor::repair_block()
{
    for (auto tokenIndex = _tokenPartition.startIndex; tokenIndex <= _tokenPartition.endIndex; ++tokenIndex) {
        auto token = _cluster->tokenPointers[tokenIndex];
        token->repair();
    }
    __syncthreads();
}

__inline__ __device__ void TokenProcessor::createCellFunctionData_block()
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
/ *
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
* /
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

__inline__ __device__ void TokenProcessor::resetTags_block()
{
    for (auto cellIndex = _cellPartition.startIndex; cellIndex <= _cellPartition.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        if (1 == cell->alive) {
            cell->tag = 0;
        }
    }
    __syncthreads();
}
*/
