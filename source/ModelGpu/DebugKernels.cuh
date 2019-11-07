#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "DEBUG_cluster.cuh"

__global__ void DEBUG_checkCluster(SimulationData data, int numClusters, int parameter)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto const clusterPointer = &data.entities.clusterPointerArrays.getArray(0).at(clusterIndex);
        DEBUG_cluster::check_blockCall(&data, *clusterPointer, parameter);
    }
}
