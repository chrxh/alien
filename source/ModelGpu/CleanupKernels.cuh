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
    PartitionData pointerBlock = calcPartition(
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
        newParticlePointers = data.entitiesForCleanup.particlePointers.getNewSubarray(numParticlePointers);
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

__global__ void cleanupParticles(SimulationData data)
{
    auto& particlePointers = data.entities.particlePointers;
    PartitionData pointerBlock = calcPartition(particlePointers.getNumEntries(),
        threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numParticlesToCopy = pointerBlock.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = data.entitiesForCleanup.particles.getNewSubarray(numParticlesToCopy);

        int newParticleIndex = 0;
        for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
            auto& particlePointer = particlePointers.at(index);
            auto& newParticle = newParticles[newParticleIndex];
            newParticle = *particlePointer;
            particlePointer = &newParticle;

            ++newParticleIndex;
        }
    }
}

__global__ void cleanupClusterPointers(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    auto& newClusters = data.entitiesForCleanup.clusterPointerArrays.getArray(clusterArrayIndex);
    PartitionData pointerBlock = calcPartition(clusters.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

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
    PartitionData pointerBlock = calcPartition(clusters.getNumEntries(), 
        threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numClustersToCopy = pointerBlock.numElements();
    if (numClustersToCopy > 0) {
        Cluster* newClusters = data.entitiesForCleanup.clusters.getNewSubarray(numClustersToCopy);
        
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
    PartitionData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Cell** newCellPointers;
        if (0 == threadIdx.x) {
            newCellPointers = data.entitiesForCleanup.cellPointers.getNewSubarray(cluster->numCellPointers);
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
    PartitionData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Cell* newCells;
        if (0 == threadIdx.x) {
            newCells = data.entitiesForCleanup.cells.getNewSubarray(cluster->numCellPointers);
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

        PartitionData tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto token = cluster->tokenPointers[tokenIndex];
            int relCellIndex = token->cell->tag;
            token->cell = newCells + relCellIndex;

            relCellIndex = token->sourceCell->tag;
            token->sourceCell = newCells + relCellIndex;
        }
        __syncthreads();
    }
}

__global__ void cleanupTokenPointers(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    PartitionData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Token** newTokenPointers;
        if (0 == threadIdx.x) {
            newTokenPointers = data.entitiesForCleanup.tokenPointers.getNewSubarray(cluster->numTokenPointers);
        }
        __syncthreads();

        auto tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto& origToken = cluster->tokenPointers[tokenIndex];
            newTokenPointers[tokenIndex] = origToken;
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            cluster->tokenPointers = newTokenPointers;
        }
        __syncthreads();
    }
}

__global__ void cleanupTokens(SimulationData data, int clusterArrayIndex)
{
    auto& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    PartitionData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusters.at(clusterIndex);

        __shared__ Token* newTokens;
        if (0 == threadIdx.x) {
            newTokens = data.entitiesForCleanup.tokens.getNewSubarray(cluster->numTokenPointers);
        }
        __syncthreads();

        auto tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);

        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto& origTokenPointer = cluster->tokenPointers[tokenIndex];
            auto& newToken = newTokens[tokenIndex];
            newToken = *origTokenPointer;
            origTokenPointer = &newTokens[tokenIndex];
        }
        __syncthreads();
    }
}

__global__ void cleanupCellMap(SimulationData data)
{
    data.cellMap.cleanup_gridCall();
}

__global__ void cleanupParticleMap(SimulationData data)
{
    data.particleMap.cleanup_gridCall();
}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cleanup(SimulationData data)
{
    data.entitiesForCleanup.clusterPointerArrays.reset();
    for (int i = 0; i < cudaConstants.NUM_CLUSTERPOINTERARRAYS; ++i) {
        cleanupClusterPointers<<<cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK>>>(data, i);
    }
    cudaDeviceSynchronize();
    data.entities.clusterPointerArrays.swapArrays(data.entitiesForCleanup.clusterPointerArrays);

    data.entitiesForCleanup.particlePointers.reset();
    cleanupParticlePointers << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    data.entities.particlePointers.swapArray(data.entitiesForCleanup.particlePointers);

    if (data.entities.particles.getNumEntries() > cudaConstants.MAX_PARTICLES * FillLevelFactor) {
        data.entitiesForCleanup.particles.reset();
        cleanupParticles << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
        cudaDeviceSynchronize();
        data.entities.particles.swapArray(data.entitiesForCleanup.particles);
    }

    if (data.entities.clusters.getNumEntries() > cudaConstants.MAX_CLUSTERS * FillLevelFactor) {
        data.entitiesForCleanup.clusters.reset();
        for (int i = 0; i < cudaConstants.NUM_CLUSTERPOINTERARRAYS; ++i) {
            cleanupClusters << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> >(data, i);
        }
        cudaDeviceSynchronize();
        data.entities.clusters.swapArray(data.entitiesForCleanup.clusters);
    }

    if (data.entities.cellPointers.getNumEntries() > cudaConstants.MAX_CELLPOINTERS * FillLevelFactor) {
        data.entitiesForCleanup.cellPointers.reset();
        KERNEL_CALL(cleanupCellPointers, cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK, data);
        data.entities.cellPointers.swapArray(data.entitiesForCleanup.cellPointers);
    }

    if (data.entities.cells.getNumEntries() > cudaConstants.MAX_CELLS * FillLevelFactor) {
        data.entitiesForCleanup.cells.reset();
        KERNEL_CALL(cleanupCells, cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK, data);
        data.entities.cells.swapArray(data.entitiesForCleanup.cells);
    }

    if (data.entities.tokenPointers.getNumEntries() > cudaConstants.MAX_TOKENPOINTERS * FillLevelFactor) {
        data.entitiesForCleanup.tokenPointers.reset();
        KERNEL_CALL(cleanupTokenPointers, cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK, data);
        data.entities.tokenPointers.swapArray(data.entitiesForCleanup.tokenPointers);
    }

    if (data.entities.tokens.getNumEntries() > cudaConstants.MAX_TOKENS * FillLevelFactor) {
        data.entitiesForCleanup.tokens.reset();
        KERNEL_CALL(cleanupTokens, cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK, data);
        data.entities.tokens.swapArray(data.entitiesForCleanup.tokens);
    }

    cleanupCellMap<<<cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cleanupParticleMap<< <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
}
