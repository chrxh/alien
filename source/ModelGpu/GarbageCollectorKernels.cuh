#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"

__device__ void garbageCollector(SimulationData &data, int clusterIndex)
{
    auto& cluster = data.clusters.getEntireArray()[clusterIndex];

    __shared__ Cell* newCells;
    __shared__ Cell** newCellPointers;
    if (0 == threadIdx.x) {
        newCells = data.cellsTemp.getNewSubarray(cluster.numCellPointers);
        newCellPointers = data.cellPointersTemp.getNewSubarray(cluster.numCellPointers);
    }
    __syncthreads();

    auto cellBlockData = calcPartition(cluster.numCellPointers, threadIdx.x, blockDim.x);
    for (int cellIndex = cellBlockData.startIndex; cellIndex <= cellBlockData.endIndex; ++cellIndex) {
        auto& origCell = *cluster.cellPointers[cellIndex];
        auto& newCell = newCells[cellIndex];
        newCell = origCell;
        newCellPointers[cellIndex] = &origCell;

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
