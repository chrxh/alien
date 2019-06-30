#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "Cluster.cuh"
#include "Cell.cuh"
#include "Token.cuh"

namespace {
    __device__ const float FillLevelFactor = 2.0f / 3.0f;
}

__global__ void cleanupParticlePointers(SimulationData data)
{
    auto& particlePointers = data.entities.particlePointers;
    BlockData pointerBlock = calcPartition(
        particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    __shared__ int numParticlePointers;
    if (0 == threadIdx.x) {
        numParticlePointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        if (particlePointers.at(index) != nullptr) {
            atomicAdd(&numParticlePointers, 1);
        }
    }
    __syncthreads();

    __shared__ Particle** newParticlePointers;
    if (0 == threadIdx.x) {
        newParticlePointers = data.entitiesNew.particlePointers.getNewSubarray(numParticlePointers);
        numParticlePointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        auto const& particlePointer = particlePointers.at(index);
        if (particlePointer != nullptr) {
            int newIndex = atomicAdd(&numParticlePointers, 1);
            newParticlePointers[newIndex] = particlePointer;
        }
    }
    __syncthreads();
}


__global__ void cleanupClusterPointers(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    auto& newClusters = data.entitiesNew.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData pointerBlock = calcPartition(clusters.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    __shared__ int numClusterPointers;
    if (0 == threadIdx.x) {
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        if (clusters.at(index) != nullptr) {
            atomicAdd(&numClusterPointers, 1);
        }
    }
    __syncthreads();

    __shared__ Cluster** newClusterPointers;
    if (0 == threadIdx.x) {
        newClusterPointers = newClusters.getNewSubarray(numClusterPointers);
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        auto const& clusterPointer = clusters.at(index);
        if (clusterPointer != nullptr) {
            int newIndex = atomicAdd(&numClusterPointers, 1);
            newClusterPointers[newIndex] = clusterPointer;
        }
    }
    __syncthreads();
}

__global__ void cleanupClusters(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData pointerBlock = calcPartition(clusters.getNumEntries(), 
        threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numClustersToCopy = pointerBlock.numElements();
    if (numClustersToCopy > 0) {
        Cluster* newClusters = data.entitiesNew.clusters.getNewSubarray(numClustersToCopy);
        
        int newClusterIndex = 0;
        for (int clusterIndex = pointerBlock.startIndex; clusterIndex <= pointerBlock.endIndex; ++clusterIndex) {
            auto& clusterPointer = clusters.at(clusterIndex);
            auto& newCluster = newClusters[newClusterIndex];
            newCluster = *clusterPointer;
            clusterPointer = &newCluster;
            for (int cellIndex = 0; cellIndex < newCluster.numCellPointers; ++cellIndex) {
                auto& cell = newCluster.cellPointers[cellIndex];
                cell->cluster = &newCluster;
            }

            ++newClusterIndex;
        }
    }
}

__global__ void cleanupCellPointers(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Cell** newCellPointers;
        if (0 == threadIdx.x) {
            newCellPointers = data.entitiesNew.cellPointers.getNewSubarray(cluster->numCellPointers);
        }
        __syncthreads();


        auto cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);

        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto& origCell = cluster->cellPointers[cellIndex];
            newCellPointers[cellIndex] = origCell;
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            cluster->cellPointers = newCellPointers;
        }
        __syncthreads();
    }
}

__global__ void cleanupCells(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Cell* newCells;
        if (0 == threadIdx.x) {
            newCells = data.entitiesNew.cells.getNewSubarray(cluster->numCellPointers);
        }
        __syncthreads();

        auto cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        cluster->tagCellByIndex_blockCall(cellBlock);

        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto& origCellPtr = cluster->cellPointers[cellIndex];
            auto& newCell = newCells[cellIndex];
            newCell = *origCellPtr;
            origCellPtr = &newCells[cellIndex];

            for (int i = 0; i < newCell.numConnections; ++i) {
                auto relCellIndex = origCellPtr->connections[i]->tag;
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
}

__device__ void cleanupClusterOnMap(SimulationData &data, int clusterArrayIndex, int clusterIndex)
{
    auto const& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    auto const& cluster = clusters.at(clusterIndex);

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

__global__ void cleanupClustersOnMap(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);

    BlockData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        cleanupClusterOnMap(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void cleanupParticlesOnMap(SimulationData data)
{
    BlockData particleBlock =
        calcPartition(data.entitiesNew.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
        cleanupParticleOnMap(data, particleIndex);
    }
}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cleanup(SimulationData data)
{
    data.entitiesNew.clusterPointerArrays.reset();
    for (int i = 0; i < NUM_CLUSTERPOINTERARRAYS; ++i) {
        cleanupClusterPointers<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(data, i);
    }
    cudaDeviceSynchronize();
    data.entities.clusterPointerArrays.swapArrays(data.entitiesNew.clusterPointerArrays);

    data.entitiesNew.particlePointers.reset();
    cleanupParticlePointers << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    data.entities.particlePointers.swapArray(data.entitiesNew.particlePointers);

    if (data.entities.clusters.getNumEntries() > MAX_CLUSTERS * FillLevelFactor) {
        data.entitiesNew.clusters.reset();
        for (int i = 0; i < NUM_CLUSTERPOINTERARRAYS; ++i) {
            cleanupClusters << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> >(data, i);
        }
        cudaDeviceSynchronize();
        data.entities.clusters.swapArray(data.entitiesNew.clusters);
    }

    if (data.entities.cellPointers.getNumEntries() > MAX_CELLPOINTERS * FillLevelFactor) {
        data.entitiesNew.cellPointers.reset();
        MULTI_CALL(cleanupCellPointers, data);
        data.entities.cellPointers.swapArray(data.entitiesNew.cellPointers);
    }

    if (data.entities.cells.getNumEntries() > MAX_CELLS * FillLevelFactor) {
        data.entitiesNew.cells.reset();
        MULTI_CALL(cleanupCells, data);
        data.entities.cells.swapArray(data.entitiesNew.cells);
    }

    MULTI_CALL(cleanupClustersOnMap, data);
    cleanupParticlesOnMap<< <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
}
