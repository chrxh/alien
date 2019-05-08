#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class TokenProcessorOnCopyData
{
public:
	__inline__ __device__ void init(SimulationData& data, int clusterIndex);

	__inline__ __device__ void processingTokenSpreading();

private:
	__inline__ __device__ void copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell);

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
}

__inline__ __device__ void TokenProcessorOnCopyData::processingTokenSpreading()
{
	for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
		Cell& cell = _cluster->cells[cellIndex];
		if (cell.alive) {
			cell.tag = 0;
		}
	}
	__shared__ int anticipatedTokens;
	if (0 == threadIdx.x) {
		anticipatedTokens = 0;
	}
	__syncthreads();

    Cell* candidateCellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    Cell* cellsForEnergyAveraging[MAX_CELL_BONDS + 1];
    for (int tokenIndex = _startTokenIndex; tokenIndex <= _endTokenIndex; ++tokenIndex) {
        Token const& token = _cluster->tokens[tokenIndex];
        Cell& cell = *token.cell;
        if (token.energy < cudaSimulationParameters.tokenMinEnergy) {
            continue;
        }

        int tokenBranchNumber = token.memory[0];
        int numCandidateCellsForEnergyAveraging = 1;
        candidateCellsForEnergyAveraging[0] = &cell;
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
            candidateCellsForEnergyAveraging[numCandidateCellsForEnergyAveraging++] = &connectingCell;
            atomicAdd(&anticipatedTokens, 1);
        }

        float averageEnergy = 0;
        int numCellsForEnergyAveraging = 0;
        for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
            Cell* cell = candidateCellsForEnergyAveraging[index];
            float origCellEnergy = atomicExch(&cell->energy, -1.0f);    //-1 is temporary value marking the cell for the moment
            if (origCellEnergy != -1.0f) {
                averageEnergy += origCellEnergy;
                cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
            }
        }
        averageEnergy /= numCellsForEnergyAveraging;
        for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
            Cell* cell = cellsForEnergyAveraging[index];
            atomicExch(&cell->energy, averageEnergy);
        }
    }
	__syncthreads();

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

__inline__ __device__ void TokenProcessorOnCopyData::copyToken(Token const* sourceToken, Token* targetToken, Cell* targetCell)
{
	*targetToken = *sourceToken;
	targetToken->memory[0] = targetCell->branchNumber;
	targetToken->cell = targetCell;
}
