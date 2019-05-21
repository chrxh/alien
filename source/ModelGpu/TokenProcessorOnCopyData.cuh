#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"
#include "CellComputerFunction.cuh"
#include "EnergyGuidance.cuh"

class TokenProcessorOnCopyData
{
public:
	__inline__ __device__ void init(SimulationData& data, int clusterIndex);

	__inline__ __device__ void processingSpreadingAndFeatures();

private:
    __inline__ __device__ void calcAnticipatedTokens(int& result);
	__inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);

    __inline__ __device__ void processingCellFeatures(Cell const* sourceCell, Token* token);


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

__inline__ __device__ void TokenProcessorOnCopyData::init(SimulationData & data, int clusterIndex)
{
	_data = &data;
	_cluster = &data.clustersAC1.getEntireArray()[clusterIndex];

	calcPartition(_cluster->numCells, threadIdx.x, blockDim.x, _startCellIndex, _endCellIndex);
	calcPartition(_cluster->numTokens, threadIdx.x, blockDim.x, _startTokenIndex, _endTokenIndex);
    int dummy = 0;
    ++dummy;
}

__inline__ __device__ void TokenProcessorOnCopyData::processingSpreadingAndFeatures()
{
    if (0 == _cluster->numTokens) {
        return;
    }

    __shared__ int anticipatedTokens;
    calcAnticipatedTokens(anticipatedTokens);

	__shared__ Token* newTokens;
	__shared__ int newNumTokens;
	if (0 == threadIdx.x) {
		newTokens = _data->tokensAC2.getNewSubarray(anticipatedTokens);
		newNumTokens = 0;
	}
	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell& cell = _cluster->cells[cellIndex];
		if (cell.alive) {
			cell.tag = 0;
		}
	}
	__syncthreads();
    
	for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
		Token& token = _cluster->tokens[tokenIndex];
		Cell& cell = *token.cell;
		if (token.energy < cudaSimulationParameters.tokenMinEnergy) {
			continue;
		}

		int tokenBranchNumber = token.memory[0];

		int numFreePlaces = 0;
		for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
			Cell const& connectingCell = *cell.connections[connectionIndex];
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
			++numFreePlaces;
		}

		if (0 == numFreePlaces) {
			atomicAdd(&cell.energy, token.energy);
			continue;
		}

		float availableTokenEnergyForCell = token.energy / numFreePlaces;
		float remainingTokenEnergy = token.energy;
		for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
			Cell& connectingCell = *cell.connections[connectionIndex];
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

			int tokenIndex = atomicAdd(&newNumTokens, 1);
			Token& newToken = newTokens[tokenIndex];
			copyToken(&token, &newToken, &connectingCell);
            processingCellFeatures(&cell, &newToken);

            if (token.energy - availableTokenEnergyForCell > 0) {
                auto origConnectingCellEnergy = atomicAdd(&connectingCell.energy, -(token.energy - availableTokenEnergyForCell));
                if (origConnectingCellEnergy > cudaSimulationParameters.cellMinEnergy + token.energy - availableTokenEnergyForCell) {
                    newToken.energy = token.energy;
                }
                else {
                    atomicAdd(&connectingCell.energy, token.energy - availableTokenEnergyForCell);
                    newToken.energy = availableTokenEnergyForCell;
                }
            }
			remainingTokenEnergy -= availableTokenEnergyForCell;
		}
		if (remainingTokenEnergy > 0) {
			atomicAdd(&cell.energy, remainingTokenEnergy);
		}
	}
	__syncthreads();


	if (0 == threadIdx.x) {
		_cluster->tokens = newTokens;
		_cluster->numTokens = newNumTokens;
	}
	__syncthreads();
}

__inline__ __device__ void TokenProcessorOnCopyData::calcAnticipatedTokens(int& result)
{
    if (0 == threadIdx.x) {
        result = 0;
    }
    __syncthreads();

    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        Token const& token = _cluster->tokens[tokenIndex];
        Cell& cell = *token.cell;
        if (token.energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token.memory[0];
        for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
            Cell& connectingCell = *cell.connections[connectionIndex];
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
            atomicAdd(&result, 1);
        }
    }
    __syncthreads();

}

__inline__ __device__ void TokenProcessorOnCopyData::copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell)
{
	*targetToken = *sourceToken;
	targetToken->memory[0] = targetCell->branchNumber;
	targetToken->cell = targetCell;
}

__inline__ __device__ void TokenProcessorOnCopyData::processingCellFeatures(Cell const* sourceCell, Token * token)
{
    auto cell = token->cell;
    int locked;
    do {
        locked = atomicExch(&cell->locked, 1);
        if (0 == locked) {
            EnergyGuidance::processing(token);
            auto type = static_cast<Enums::CellFunction::Type>(
                cell->cellFunctionType % static_cast<int>(Enums::CellFunction::_COUNTER));
            switch (type) {
            case Enums::CellFunction::COMPUTER: {
                CellComputerFunction::processing(token);
            } break;
            }
            atomicExch(&cell->locked, 0);
        }
    } while (1 == locked);

}

