#pragma once

#include "Base.cuh"
#include "CudaAccessTOs.cuh"
#include "Map.cuh"
#include "ModelBasic/ElementaryTypes.h"
#include "Physics.cuh"
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

class TokenProcessorOnOrigData
{
public:
    __inline__ __device__ void init(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingEnergyAveraging();

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
__inline__ __device__ void TokenProcessorOnOrigData::init(SimulationData& data, int clusterIndex)
{
    _data = &data;
    _cluster = &data.clusters.getEntireArray()[clusterIndex];

    calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x, _startCellIndex, _endCellIndex);
    calcPartition(_cluster->numTokens, threadIdx.x, blockDim.x, _startTokenIndex, _endTokenIndex);
}

__inline__ __device__ void TokenProcessorOnOrigData::processingEnergyAveraging()
{
    if (0 == _cluster->numTokens) {
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
        }

        float averageEnergy = 0;
        int numCellsForEnergyAveraging = 0;
        for (int index = 0; index < numCandidateCellsForEnergyAveraging; ++index) {
            Cell* cell = candidateCellsForEnergyAveraging[index];
            int locked = atomicExch(&cell->locked, 1);
            if (0 == locked) {
                averageEnergy += cell->energy;
                cellsForEnergyAveraging[numCellsForEnergyAveraging++] = cell;
            }
        }
        averageEnergy /= numCellsForEnergyAveraging;
        for (int index = 0; index < numCellsForEnergyAveraging; ++index) {
            Cell* cell = cellsForEnergyAveraging[index];
            cell->energy = averageEnergy;
            atomicExch(&cell->locked, 0);
        }
    }
    __syncthreads();
}
