#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CudaMonitorData.cuh"

/************************************************************************/
/* Helpers    															*/
/************************************************************************/

__global__ void
getMonitorDataForClusters(Array<Cluster*> clusterPointers, CudaMonitorData monitorData)
{
/*
    auto const clusterPartition =
        calcPartition(clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto const cluster = clusterPointers.at(clusterIndex);
        if (nullptr == cluster) {
            continue;
        }
        auto const mass = static_cast<float>(cluster->numCellPointers);
        if (0 == threadIdx.x) {
            monitorData.incNumClusters(1);
            monitorData.incNumCells(cluster->numCellPointers);
            monitorData.incNumTokens(cluster->numTokenPointers);
            if (cluster->numTokenPointers > 0) {
                monitorData.incNumClustersWithTokens(1);
            }
        }

        __shared__ float clusterInternalEnergy;
        if (0 == threadIdx.x) {
            clusterInternalEnergy = 0.0f;
        }
        __syncthreads();

        auto const cellPartition =
            calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellPartition.startIndex; cellIndex <= cellPartition.endIndex; ++cellIndex) {
            auto const cell = cluster->cellPointers[cellIndex];
            atomicAdd_block(&clusterInternalEnergy, cell->getEnergy());
        }
        auto const tokenPartition =
            calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (auto tokenIndex = tokenPartition.startIndex; tokenIndex <= tokenPartition.endIndex; ++tokenIndex) {
            auto const token = cluster->tokenPointers[tokenIndex];
            atomicAdd_block(&clusterInternalEnergy, token->getEnergy());
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            monitorData.incInternalEnergy(clusterInternalEnergy);
        }
        __syncthreads();

    }
*/
}

__global__ void getMonitorDataForParticles(SimulationData data, CudaMonitorData monitorData)
{
    if (0 == threadIdx.x && 0 == blockIdx.x) {
        monitorData.incNumParticles(data.entities.particlePointers.getNumEntries());
    }

    auto const partition = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (auto index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const particle = data.entities.particlePointers.at(index);
        monitorData.incInternalEnergy(particle->energy);
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void cudaGetCudaMonitorData(SimulationData data, CudaMonitorData monitorData)
{
    monitorData.reset();

/*
    KERNEL_CALL(getMonitorDataForClusters, data.entities.clusterPointers, monitorData);
    KERNEL_CALL(getMonitorDataForParticles, data, monitorData);
*/
}

