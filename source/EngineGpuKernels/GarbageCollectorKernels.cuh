#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "Cell.cuh"
#include "Token.cuh"

__global__ void preparePointerArraysForCleanup(SimulationData data)
{
    data.entitiesForCleanup.particlePointers.reset();
    data.entitiesForCleanup.cellPointers.reset();
    data.entitiesForCleanup.tokenPointers.reset();
}

__global__ void prepareArraysForCleanup(SimulationData data)
{
    data.entitiesForCleanup.particles.reset();
    data.entitiesForCleanup.cells.reset();
    data.entitiesForCleanup.tokens.reset();
}

template<typename Entity>
__global__ void cleanupPointerArray(Array<Entity> entityArray, Array<Entity> newEntityArray)
{
    auto partition =
        calcPartition(entityArray.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    __shared__ int numEntities;
    if (0 == threadIdx.x) {
        numEntities = 0;
    }
    __syncthreads();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        if (entityArray.at(index) != nullptr) {
            atomicAdd_block(&numEntities, 1);
        }
    }
    __syncthreads();

    __shared__ Entity* newEntities;
    if (0 == threadIdx.x) {
        if (numEntities > 0) {
            newEntities = newEntityArray.getNewSubarray(numEntities);
        }
        numEntities = 0;
    }
    __syncthreads();

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& entity = entityArray.at(index);
        if (entity != nullptr) {
            int newIndex = atomicAdd(&numEntities, 1);
            newEntities[newIndex] = entity;
        }
    }
    __syncthreads();
}

__global__ void
cleanupParticles(Array<Particle*> particlePointers, Array<Particle> particles)
{
    //assumes that particlePointers are already cleaned up
    PartitionData pointerBlock = calcPartition(particlePointers.getNumEntries(),
        threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numParticlesToCopy = pointerBlock.numElements();
    if (numParticlesToCopy > 0) {
        auto newParticles = particles.getNewSubarray(numParticlesToCopy);

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

__global__ void cleanupCellsStep1(Array<Cell*> cellPointers, Array<Cell> cells)
{
    //assumes that cellPointers are already cleaned up
    PartitionData pointerBlock =
        calcPartition(cellPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    int numCellsToCopy = pointerBlock.numElements();
    if (numCellsToCopy > 0) {
        auto newCells = cells.getNewSubarray(numCellsToCopy);

        int newCellIndex = 0;
        for (int index = pointerBlock.startIndex; index <= pointerBlock.endIndex; ++index) {
            auto& cellPointer = cellPointers.at(index);
            auto& newCell = newCells[newCellIndex];
            newCell = *cellPointer;

            cellPointer->tag = &newCell - cells.getArray();    //save index of new cell in old cell
            cellPointer = &newCell;

            ++newCellIndex;
        }
    }
}

__global__ void cleanupCellsStep2(Array<Token*> tokenPointers, Array<Cell> cells)
{
    {
        auto partition =
            calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            for (int i = 0; i < cell.numConnections; ++i) {
                auto& connectedCell = cell.connections[i].cell;
                cell.connections[i].cell = &cells.at(connectedCell->tag);
            }
        }
    }
    {
        auto partition =
            calcPartition(tokenPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            if (auto& token = tokenPointers.at(index)) {
                token->cell = &cells.at(token->cell->tag);
                token->sourceCell = &cells.at(token->sourceCell->tag);
            }
        }
    }
}

__global__ void cleanupTokens(Array<Token*> tokenPointers, Array<Token> newToken)
{
    auto partition =
        calcPartition(tokenPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    if (partition.numElements() > 0) {
        Token* newEntities = newToken.getNewSubarray(partition.numElements());

        int targetIndex = 0;
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& token = tokenPointers.at(index);
            newEntities[targetIndex] = *token;
            token = &newEntities[targetIndex];
            ++targetIndex;
        }
    }
}

__global__ void cleanupCellMap(SimulationData data)
{
    data.cellMap.cleanup_system();
}

__global__ void cleanupParticleMap(SimulationData data)
{
    data.particleMap.cleanup_system();
}

__global__ void swapPointerArrays(SimulationData data)
{
    data.entities.particlePointers.swapContent(data.entitiesForCleanup.particlePointers);
    data.entities.cellPointers.swapContent(data.entitiesForCleanup.cellPointers);
    data.entities.tokenPointers.swapContent(data.entitiesForCleanup.tokenPointers);
}

__global__ void swapArrays(SimulationData data)
{
    data.entities.cells.swapContent(data.entitiesForCleanup.cells);
    data.entities.tokens.swapContent(data.entitiesForCleanup.tokens);
    data.entities.particles.swapContent(data.entitiesForCleanup.particles);
}

/*
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
*/

/************************************************************************/
/* Main                                                                 */
/************************************************************************/

__global__ void cleanupEntityArraysNecessary(SimulationData data, bool* result)
{
    if (data.entities.particles.getNumEntries() > data.entities.particles.getSize() * Const::ArrayFillLevelFactor
        || data.entities.cells.getNumEntries() > data.entities.cells.getSize() * Const::ArrayFillLevelFactor
        || data.entities.tokens.getNumEntries() > data.entities.tokens.getSize() * Const::ArrayFillLevelFactor) {
        *result = true;
    } else {
        *result = false;
    }
}

__global__ void cleanupAfterDataManipulationKernel(SimulationData data)
{
    data.entitiesForCleanup.particlePointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(
        cleanupPointerArray<Particle*>, data.entities.particlePointers, data.entitiesForCleanup.particlePointers);
    data.entities.particlePointers.swapContent(data.entitiesForCleanup.particlePointers);

    data.entitiesForCleanup.cellPointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupPointerArray<Cell*>, data.entities.cellPointers, data.entitiesForCleanup.cellPointers);
    data.entities.cellPointers.swapContent(data.entitiesForCleanup.cellPointers);

    data.entitiesForCleanup.tokenPointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupPointerArray<Token*>, data.entities.tokenPointers, data.entitiesForCleanup.tokenPointers);
    data.entities.tokenPointers.swapContent(data.entitiesForCleanup.tokenPointers);

    data.entitiesForCleanup.particles.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupParticles, data.entities.particlePointers, data.entitiesForCleanup.particles);
    data.entities.particles.swapContent(data.entitiesForCleanup.particles);

    data.entitiesForCleanup.cells.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupCellsStep1, data.entities.cellPointers, data.entitiesForCleanup.cells);
    DEPRECATED_KERNEL_CALL_SYNC(cleanupCellsStep2, data.entities.tokenPointers, data.entitiesForCleanup.cells);
    data.entities.cells.swapContent(data.entitiesForCleanup.cells);

    data.entitiesForCleanup.tokens.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupTokens, data.entities.tokenPointers, data.entitiesForCleanup.tokens);
    data.entities.tokens.swapContent(data.entitiesForCleanup.tokens);

    data.entitiesForCleanup.strings.reset();
/*
    KERNEL_CALL(cleanupMetadata, data.entities.clusterPointers, data.entitiesForCleanup.strings);
    data.entities.strings.swapContent(data.entitiesForCleanup.strings);
*/
}

__global__ void cudaCopyEntities(SimulationData data)
{
    data.entitiesForCleanup.particlePointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupPointerArray<Particle*>, data.entities.particlePointers, data.entitiesForCleanup.particlePointers);

    data.entitiesForCleanup.cellPointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupPointerArray<Cell*>, data.entities.cellPointers, data.entitiesForCleanup.cellPointers);

    data.entitiesForCleanup.tokenPointers.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupPointerArray<Token*>, data.entities.tokenPointers, data.entitiesForCleanup.tokenPointers);

    data.entitiesForCleanup.particles.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupParticles, data.entitiesForCleanup.particlePointers, data.entitiesForCleanup.particles);

    data.entitiesForCleanup.cells.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupCellsStep1, data.entitiesForCleanup.cellPointers, data.entitiesForCleanup.cells);
    DEPRECATED_KERNEL_CALL_SYNC(cleanupCellsStep2, data.entitiesForCleanup.tokenPointers, data.entitiesForCleanup.cells);

    data.entitiesForCleanup.tokens.reset();
    DEPRECATED_KERNEL_CALL_SYNC(cleanupTokens, data.entitiesForCleanup.tokenPointers, data.entitiesForCleanup.tokens);
}
