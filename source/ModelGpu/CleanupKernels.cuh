#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "Cluster.cuh"
#include "Cell.cuh"
#include "Token.cuh"
#include "FreezingKernels.cuh"

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

__global__ void cleanupClusterPointers(Array<Cluster*> clusterPointers, Array<Cluster*> clusterPointersForCleanup)
{
    PartitionData pointerBlock = calcPartition(clusterPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    __shared__ int numClusterPointers;
    if (0 == threadIdx.x) {
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        if (clusterPointers.at(index) != nullptr) {
            atomicAdd(&numClusterPointers, 1);
        }
    }
    __syncthreads();

    __shared__ Cluster** newClusterPointers;
    if (0 == threadIdx.x) {
        newClusterPointers = clusterPointersForCleanup.getNewSubarray(numClusterPointers);
        numClusterPointers = 0;
    }
    __syncthreads();

    for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
        auto const& clusterPointer = clusterPointers.at(index);
        if (clusterPointer != nullptr) {
            int newIndex = atomicAdd(&numClusterPointers, 1);
            newClusterPointers[newIndex] = clusterPointer;
        }
    }
    __syncthreads();
}

__global__ void cleanupClusters(Array<Cluster*> clusterPointers, Array<Cluster> clustersForCleanup)
{
    PartitionData pointerBlock = calcPartition(clusterPointers.getNumEntries(), 
        threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numClustersToCopy = pointerBlock.numElements();
    if (numClustersToCopy > 0) {
        Cluster* newClusters = clustersForCleanup.getNewSubarray(numClustersToCopy);
        
        int newClusterIndex = 0;
        for (int clusterIndex = pointerBlock.startIndex; clusterIndex <= pointerBlock.endIndex; ++clusterIndex) {
            auto& clusterPointer = clusterPointers.at(clusterIndex);
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

__global__ void cleanupCellPointers(Array<Cluster*> clusterPointers, Array<Cell*> cellPointers)
{
    PartitionData clusterBlock = calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusterPointers.at(clusterIndex);

        __shared__ Cell** newCellPointers;
        if (0 == threadIdx.x) {
            newCellPointers = cellPointers.getNewSubarray(cluster->numCellPointers);
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

__global__ void cleanupCells(Array<Cluster*> clusterPointers, Array<Cell> cells)
{
    PartitionData clusterBlock = calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusterPointers.at(clusterIndex);

        __shared__ Cell* newCells;
        if (0 == threadIdx.x) {
            newCells = cells.getNewSubarray(cluster->numCellPointers);
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

__global__ void cleanupTokenPointers(Array<Cluster*> clusterPointers, Array<Token*> tokenPointers)
{
    PartitionData clusterBlock = calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusterPointers.at(clusterIndex);

        __shared__ Token** newTokenPointers;
        if (0 == threadIdx.x) {
            newTokenPointers = tokenPointers.getNewSubarray(cluster->numTokenPointers);
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

__global__ void cleanupTokens(Array<Cluster*> clusterPointers, Array<Token> tokens)
{
    PartitionData clusterBlock = calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusterPointers.at(clusterIndex);

        __shared__ Token* newTokens;
        if (0 == threadIdx.x) {
            newTokens = tokens.getNewSubarray(cluster->numTokenPointers);
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

__global__ void cleanupMetadata(Array<Cluster*> clusterPointers, DynamicMemory strings)
{
    auto const clusterBlock = calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto& cluster = clusterPointers.at(clusterIndex);

        if (0 == threadIdx.x) {
            auto const len = cluster->metadata.nameLen;
            auto newName = strings.getArray<char>(len);
            for(int i = 0; i < len; ++i) {
                newName[i] = cluster->metadata.name[i];
            }
            cluster->metadata.name = newName;
        }

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto& cell = cluster->cellPointers[cellIndex];
            {
                auto const len = cell->metadata.nameLen;
                auto newName = strings.getArray<char>(len);
                for (int i = 0; i < len; ++i) {
                    newName[i] = cell->metadata.name[i];
                }
                cell->metadata.name = newName;
            }
            {
                auto const len = cell->metadata.descriptionLen;
                auto newDescription = strings.getArray<char>(len);
                for (int i = 0; i < len; ++i) {
                    newDescription[i] = cell->metadata.description[i];
                }
                cell->metadata.description = newDescription;
            }
            {
                auto const len = cell->metadata.sourceCodeLen;
                auto newSourceCode = strings.getArray<char>(len);
                for (int i = 0; i < len; ++i) {
                    newSourceCode[i] = cell->metadata.sourceCode[i];
                }
                cell->metadata.sourceCode = newSourceCode;
            }
        }
    }
}

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cleanupAfterSimulation(SimulationData data)
{
    if ((data.timestep % 5) == 0) {
        KERNEL_CALL_1_1(unfreeze, data);
    }

    KERNEL_CALL(cleanupCellMap, data);  //should be called before cleanupClusters and cleanupCells due to freezing
    KERNEL_CALL(cleanupParticleMap, data);

    data.entitiesForCleanup.clusterPointers.reset();
    KERNEL_CALL(cleanupClusterPointers, data.entities.clusterPointers, data.entitiesForCleanup.clusterPointers);
    data.entities.clusterPointers.swapContent(data.entitiesForCleanup.clusterPointers);

    data.entitiesForCleanup.particlePointers.reset();
    KERNEL_CALL(cleanupParticlePointers, data);
    data.entities.particlePointers.swapContent(data.entitiesForCleanup.particlePointers);

    if ((data.timestep % 5) == 0) {

        if (data.entities.particles.getNumEntries() > cudaConstants.MAX_PARTICLES * FillLevelFactor) {
            data.entitiesForCleanup.particles.reset();
            KERNEL_CALL(cleanupParticles, data);
            data.entities.particles.swapContent(data.entitiesForCleanup.particles);
        }

        if (data.entities.clusters.getNumEntries() > cudaConstants.MAX_CLUSTERS * FillLevelFactor) {
            data.entitiesForCleanup.clusters.reset();
            KERNEL_CALL(cleanupClusters, data.entities.clusterPointers, data.entitiesForCleanup.clusters);
            data.entities.clusters.swapContent(data.entitiesForCleanup.clusters);
        }

        if (data.entities.cellPointers.getNumEntries() > cudaConstants.MAX_CELLPOINTERS * FillLevelFactor) {
            data.entitiesForCleanup.cellPointers.reset();
            KERNEL_CALL(cleanupCellPointers, data.entities.clusterPointers, data.entitiesForCleanup.cellPointers);
            data.entities.cellPointers.swapContent(data.entitiesForCleanup.cellPointers);
        }

        if (data.entities.cells.getNumEntries() > cudaConstants.MAX_CELLS * FillLevelFactor) {
            data.entitiesForCleanup.cells.reset();
            KERNEL_CALL(cleanupCells, data.entities.clusterPointers, data.entitiesForCleanup.cells);
            data.entities.cells.swapContent(data.entitiesForCleanup.cells);
        }

        if (data.entities.tokenPointers.getNumEntries() > cudaConstants.MAX_TOKENPOINTERS * FillLevelFactor) {
            data.entitiesForCleanup.tokenPointers.reset();
            KERNEL_CALL(cleanupTokenPointers, data.entities.clusterPointers, data.entitiesForCleanup.tokenPointers);
            data.entities.tokenPointers.swapContent(data.entitiesForCleanup.tokenPointers);
        }

        if (data.entities.tokens.getNumEntries() > cudaConstants.MAX_TOKENS * FillLevelFactor) {
            data.entitiesForCleanup.tokens.reset();
            KERNEL_CALL(cleanupTokens, data.entities.clusterPointers, data.entitiesForCleanup.tokens);
            data.entities.tokens.swapContent(data.entitiesForCleanup.tokens);
        }

        if (data.entities.strings.getNumBytes() > cudaConstants.MAX_STRINGBYTES * FillLevelFactor) {
            data.entitiesForCleanup.strings.reset();
            KERNEL_CALL(cleanupMetadata, data.entities.clusterPointers, data.entitiesForCleanup.strings);
            data.entities.strings.swapContent(data.entitiesForCleanup.strings);
        }
    }
}

__global__ void cleanupAfterDataManipulation(SimulationData data)
{

    data.entitiesForCleanup.clusterPointers.reset();
    KERNEL_CALL(cleanupClusterPointers, data.entities.clusterPointers, data.entitiesForCleanup.clusterPointers);
    data.entities.clusterPointers.swapContent(data.entitiesForCleanup.clusterPointers);

    data.entitiesForCleanup.particlePointers.reset();
    KERNEL_CALL(cleanupParticlePointers, data);
    data.entities.particlePointers.swapContent(data.entitiesForCleanup.particlePointers);

    if (data.entities.particles.getNumEntries() > cudaConstants.MAX_PARTICLES * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.particles.reset();
        KERNEL_CALL(cleanupParticles, data);
        data.entities.particles.swapContent(data.entitiesForCleanup.particles);
    }

    if (data.entities.clusters.getNumEntries() > cudaConstants.MAX_CLUSTERS * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.clusters.reset();
        KERNEL_CALL(cleanupClusters, data.entities.clusterPointers, data.entitiesForCleanup.clusters);
        data.entities.clusters.swapContent(data.entitiesForCleanup.clusters);
    }

    if (data.entities.cellPointers.getNumEntries() > cudaConstants.MAX_CELLPOINTERS * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.cellPointers.reset();
        KERNEL_CALL(cleanupCellPointers, data.entities.clusterPointers, data.entitiesForCleanup.cellPointers);
        data.entities.cellPointers.swapContent(data.entitiesForCleanup.cellPointers);
    }

    if (data.entities.cells.getNumEntries() > cudaConstants.MAX_CELLS * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.cells.reset();
        KERNEL_CALL(cleanupCells, data.entities.clusterPointers, data.entitiesForCleanup.cells);
        data.entities.cells.swapContent(data.entitiesForCleanup.cells);
    }

    if (data.entities.tokenPointers.getNumEntries() > cudaConstants.MAX_TOKENPOINTERS * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.tokenPointers.reset();
        KERNEL_CALL(cleanupTokenPointers, data.entities.clusterPointers, data.entitiesForCleanup.tokenPointers);
        data.entities.tokenPointers.swapContent(data.entitiesForCleanup.tokenPointers);
    }

    if (data.entities.tokens.getNumEntries() > cudaConstants.MAX_TOKENS * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.tokens.reset();
        KERNEL_CALL(cleanupTokens, data.entities.clusterPointers, data.entitiesForCleanup.tokens);
        data.entities.tokens.swapContent(data.entitiesForCleanup.tokens);
    }

    if (data.entities.strings.getNumBytes() > cudaConstants.MAX_STRINGBYTES * FillLevelFactor) {
        KERNEL_CALL_1_1(unfreeze, data);

        data.entitiesForCleanup.strings.reset();
        KERNEL_CALL(cleanupMetadata, data.entities.clusterPointers, data.entitiesForCleanup.strings);
        data.entities.strings.swapContent(data.entitiesForCleanup.strings);
    }
}
