#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Entities.cuh"
#include "SimulationData.cuh"

__global__ void cudaPreparePointerArraysForCleanup(SimulationData data);
__global__ void cudaPrepareHeapForCleanup(SimulationData data);

__global__ void cudaCleanupParticles(Array<Energy*> particlePointers, Heap newHeap);
__global__ void cudaPrepareCleanupCreaturesAndGenomes(Array<Object*> cells);
__global__ void cudaCleanupGenomesStep1(Array<Object*> cells, Heap newHeap);
__global__ void cudacudaCleanupGenomesStep2(Array<Object*> cells, Heap newHeap);
__global__ void cudaCleanupCreaturesStep1(Array<Object*> cells, Heap newHeap);
__global__ void cudaCleanupCreaturesStep2(Array<Object*> cells, Heap newHeap);
__global__ void cudaCleanupCellsStep1(Array<Object*> cells, Heap newHeap);
__global__ void cudaCleanupCellsStep2(Array<Object*> cellPointers, Heap newHeap);
__global__ void cudaCleanupDependentCellData(Array<Object*> cells, Heap newHeap);
__global__ void cudaCleanupMaps(SimulationData data);
__global__ void cudaSwapPointerArrays(SimulationData data);
__global__ void cudaSwapHeaps(SimulationData data);
__global__ void cudaCheckIfCleanupIsNecessary(SimulationData data, bool* result);

template <typename Entity>
__global__ void cudaCleanupPointerArray(Array<Entity> entityArray, Array<Entity> newEntityArray)
{
    auto partition = calcSystemThreadPartition(entityArray.getNumEntries());

    __shared__ int numEntities;
    if (0 == threadIdx.x) {
        numEntities = 0;
    }
    __syncthreads();

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
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

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& entity = entityArray.at(index);
        if (entity != nullptr) {
            int newIndex = atomicAdd(&numEntities, 1);
            newEntities[newIndex] = entity;
        }
    }
    __syncthreads();
}
