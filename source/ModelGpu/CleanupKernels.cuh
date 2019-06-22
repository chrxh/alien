#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "Cluster.cuh"
#include "Cell.cuh"
#include "Token.cuh"

__device__ void cleanupCells(SimulationData &data, int clusterIndex)
{
    auto& cluster = data.entities.clusterPointers.at(clusterIndex);

    __shared__ Cell* newCells;
    if (0 == threadIdx.x) {
        newCells = data.entitiesNew.cells.getNewSubarray(cluster->numCellPointers);
    }
    __syncthreads();


    auto cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
    cluster->tagCellByIndex_blockCall(cellBlock);

    for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        auto& origCell = cluster->cellPointers[cellIndex];
        auto& newCell = newCells[cellIndex];
        newCell = *origCell;
        origCell = &newCells[cellIndex];

        for (int i = 0; i < newCell.numConnections; ++i) {
            auto relCellIndex = origCell->connections[i]->tag;
            newCell.connections[i] = newCells + relCellIndex;
        }

    }

    BlockData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
    for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto token = cluster->tokenPointers[tokenIndex];
        int relCellIndex = token->cell->tag;
        token->cell = newCells + relCellIndex;
    }
    __syncthreads();
}

__global__ void cleanupClusterPointers(SimulationData data)
{
    BlockData pointerBlock = calcPartition(data.entities.clusterPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    __shared__ int numClusterPointers;
    if (0 == threadIdx.x) {
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        if (data.entities.clusterPointers.at(index) != nullptr) {
            atomicAdd(&numClusterPointers, 1);
        }
    }
    __syncthreads();

    __shared__ Cluster** newClusterPointers;
    if (0 == threadIdx.x) {
        newClusterPointers = data.entitiesNew.clusterPointers.getNewSubarray(numClusterPointers);
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        auto const& clusterPointer = data.entities.clusterPointers.at(index);
        if (clusterPointer != nullptr) {
            int newIndex = atomicAdd(&numClusterPointers, 1);
            newClusterPointers[newIndex] = clusterPointer;
        }
    }
    __syncthreads();
}

__global__ void cleanupCellPointers(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.entities.clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = data.entities.clusters.getEntireArray()[clusterIndex];

        __shared__ Cell** newCellPointers;
        if (0 == threadIdx.x) {
            newCellPointers = data.entitiesNew.cellPointers.getNewSubarray(cluster.numCellPointers);
        }
        __syncthreads();


        auto cellBlock = calcPartition(cluster.numCellPointers, threadIdx.x, blockDim.x);

        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto& origCell = cluster.cellPointers[cellIndex];
            newCellPointers[cellIndex] = origCell;
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            cluster.cellPointers = newCellPointers;
        }
        __syncthreads();
    }
}

__device__ void cleanupClusterOnMap(SimulationData const &data, int clusterIndex)
{
    auto const &cluster = data.entitiesNew.clusterPointers.at(clusterIndex);

    int2 size = data.size;
    int numCells = cluster->numCellPointers;
    Map<Cell> map;
    map.init(size, data.cellMap);

    BlockData cellBlock = calcPartition(numCells, threadIdx.x, blockDim.x);
    for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        float2 absPos = cluster->cellPointers[cellIndex]->absPos;
        map.set(absPos, nullptr);
    }
}

__device__ void cleanupParticleOnMap(SimulationData const &data, int particleIndex)
{
    Map<Particle> map;
    map.init(data.size, data.particleMap);

    auto const &particle = data.entitiesNew.particles.getEntireArray()[particleIndex];
    map.set(particle.pos, nullptr);
}

__global__ void cleanupMaps(SimulationData data)
{

    BlockData clusterBlock = calcPartition(data.entitiesNew.clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        cleanupClusterOnMap(data, clusterIndex);
    }


    BlockData particleBlock =
        calcPartition(data.entitiesNew.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        cleanupParticleOnMap(data, particleIndex);
    }
}

__global__ void cleanup(SimulationData data)
{
    data.entitiesNew.clusterPointers.reset();

    cleanupClusterPointers << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();

    cleanupMaps << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();

    data.entities.clusterPointers.swapArrays(data.entitiesNew.clusterPointers);
}
