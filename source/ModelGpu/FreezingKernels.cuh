#pragma once

#include <cooperative_groups.h>
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "TokenProcessor.cuh"
#include "CleanupKernels.cuh"

/************************************************************************/
/* Helpers                                                              */
/************************************************************************/

__global__ void freezeClustersIfAllowed(SimulationData data)
{
    if (cudaExecutionParameters.activateFreezing) {
        auto const clusterPartition = calcPartition(
            data.entities.clusterPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
            auto& cluster = data.entities.clusterPointers.at(clusterIndex);
            if (cluster && cluster->isCandidateToFreeze()) {
                auto clusterFreezedPointer = data.entities.clusterFreezedPointers.getNewElement();
                *clusterFreezedPointer = cluster;
                cluster->freeze(clusterFreezedPointer);
                cluster = nullptr;
            }
        }
    }
}

__global__ void unfreezeAllClusters(SimulationData data)
{
    auto const clusterPartition =
        calcPartition(data.entities.clusterFreezedPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto clusterFreezed = data.entities.clusterFreezedPointers.at(clusterIndex);
        if (clusterFreezed != nullptr) {
            auto clusterPointer = data.entities.clusterPointers.getNewElement();
            *clusterPointer = clusterFreezed;
            clusterFreezed->unfreeze();
        }
    }

}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void unfreeze(SimulationData data)
{
    KERNEL_CALL(unfreezeAllClusters, data);
}

