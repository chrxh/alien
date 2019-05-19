#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "SimulationData.cuh"

__device__ void garbageCollector(SimulationData &data, int clusterIndex)
{
    auto& cluster = data.clustersAC1.getEntireArray()[clusterIndex];

    __shared__ Cell* newCells;
    if (0 == threadIdx.x) {
        newCells = data.cellsTempAC.getNewSubarray(cluster.numCells);
    }
    __syncthreads();

    int startCellIndex;
    int endCellIndex;
    calcPartition(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

    for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
        auto& origCell = cluster.cells[cellIndex];
        auto& newCell = newCells[cellIndex];
        newCell = origCell;

        for (int i = 0; i < newCell.numConnections; ++i) {
            int relIndex = origCell.connections[i] - cluster.cells;
            newCell.connections[i] = newCells + relIndex;
        }

    }

    int startTokenIndex;
    int endTokenIndex;
    calcPartition(cluster.numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);

    for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        auto& token = cluster.tokens[tokenIndex];
        int relIndex = token.cell - cluster.cells;
        token.cell = newCells + relIndex;
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        cluster.cells = newCells;
    }
    __syncthreads();
}

__global__ void garbageCollector(SimulationData data)
{
    int numEntities = data.clustersAC1.getNumEntries();

    int startIndex;
    int endIndex;
    calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
    for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
        garbageCollector(data, clusterIndex);
    }
}
