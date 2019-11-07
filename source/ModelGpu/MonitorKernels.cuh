#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CudaMonitorData.cuh"

/************************************************************************/
/* Helpers    															*/
/************************************************************************/

__global__ void
getMonitorDataForClusters(SimulationData data, CudaMonitorData monitorData, int numClusters, int clusterArrayIndex)
{
    if (0 == threadIdx.x && 0 == blockIdx.x) {
        monitorData.incNumClusters(numClusters);
    }

    auto const clusterPartition =
        calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto const cluster = data.entities.clusterPointerArrays.getArray(clusterArrayIndex).at(clusterIndex);
        auto const mass = static_cast<float>(cluster->numCellPointers);
        if (0 == threadIdx.x) {
            monitorData.incLinearKineticEnergy(Physics::linearKineticEnergy(mass, cluster->vel));
            monitorData.incRotationalKineticEnergy(Physics::rotationalKineticEnergy(cluster->angularMass, cluster->angularVel));
            monitorData.incNumCells(cluster->numCellPointers);
            monitorData.incNumTokens(cluster->numTokenPointers);
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
        monitorData.incInternalEnergy(particle->getEnergy());
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void getCudaMonitorData(SimulationData data, CudaMonitorData monitorData)
{
    monitorData.reset();

    MULTI_CALL(getMonitorDataForClusters, data, monitorData, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    getMonitorDataForParticles << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data, monitorData);
}

