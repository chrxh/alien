#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "Object.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data);
__global__ void cudaPrepareHeapForCleanup(SimulationData data);

template<typename Entity>
__global__ void cudaCleanupPointerArray(Array<Entity> entityArray, Array<Entity> newEntityArray)
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
            atomicAdd(&numEntities, 1);
        }
    }
    __syncthreads();

    __shared__ Entity* newEntities;
    if (0 == threadIdx.x) {
        if (numEntities > 0) {
            newEntities = newEntityArray.getSubArray(numEntities);
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

__global__ void cudaCleanupParticles(Array<Particle*> particlePointers, Heap rawMemory);
__global__ void cudaCleanupCellsStep1(Array<Cell*> cellPointers, Heap newHeap);
__global__ void cudaCleanupCellsStep2(Array<Cell*> cellPointers, Heap newHeap);
__global__ void cudaCleanupDependentCellData(Array<Cell*> cellPointers, Heap stringBytes);
__global__ void cudaCleanupCellMap(SimulationData data);
__global__ void cudaCleanupParticleMap(SimulationData data);
__global__ void cudaSwapPointerArrays(SimulationData data);
__global__ void cudaSwapHeaps(SimulationData data);
__global__ void cudaCheckIfCleanupIsNecessary(SimulationData data, bool* result);
