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
/* Helpers for clusters													*/
/************************************************************************/

__global__ void clusterProcessingStep1(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterIndex);
        clusterProcessor.processingMovement_blockCall();
        clusterProcessor.updateMap_blockCall();
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterIndex);
        clusterProcessor.destroyCloseCell_blockCall();
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterIndex);
        clusterProcessor
            .processingCollision_blockCall();  //attention: can result a temporarily inconsistent state, will be resolved in step 4
        clusterProcessor.processingRadiation_blockCall();
    }
}

__global__ void clusterProcessingStep4(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterIndex);
        clusterProcessor.processingCellDeath_blockCall();
        clusterProcessor.processingDecomposition_blockCall();
        clusterProcessor.processingClusterCopy_blockCall();
    }
}


/************************************************************************/
/* Helpers for tokens													*/
/************************************************************************/
__global__ void resetCellFunctionData(SimulationData data)
{
    data.cellFunctionData.mapSectionCollector.reset_gridCall();
}

__global__ void tokenProcessingStep1(SimulationData data)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.init_gridCall(data);
    tokenProcessor.processingEnergyAveraging_gridCall();
    tokenProcessor.processingSpreading_gridCall();
    tokenProcessor.processingLightWeigthedFeatures_gridCall();
}

__global__ void tokenProcessingStep2(SimulationData data, int numClusters)
{
    auto const clusterPartition = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_blockCall(data, clusterIndex);
        tokenProcessor.createCellFunctionData_blockCall();
    }
}

__global__ void tokenProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_blockCall(data, clusterIndex);
        tokenProcessor.processingHeavyWeightedFeatures_blockCall();
    }
}

/************************************************************************/
/* Helpers for particles												*/
/************************************************************************/

__global__ void particleProcessingStep1(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_gridCall(data);
    particleProcessor.processingMovement_gridCall();
    particleProcessor.updateMap_gridCall();
    particleProcessor.processingTransformation_gridCall();
}

__global__ void particleProcessingStep2(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.init_gridCall(data);
    particleProcessor.processingCollision_gridCall();
}

__global__ void particleProcessingStep3(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_gridCall(data);
    particleProcessor.processingDataCopy_gridCall();
}

/************************************************************************/
/* Freezing                                                             */
/************************************************************************/

__global__ void freezeClustersIfAllowed(SimulationData data)
{
    auto const clusterPartition = calcPartition(
        data.entities.clusterPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto& cluster = data.entities.clusterPointers.at(clusterIndex);
        if (cluster && cluster->isFreezed()) {
            auto clusterFreezedPointer = data.entities.clusterFreezedPointers.getNewElement();
            *clusterFreezedPointer = cluster;
            cluster = nullptr;
        }
    }
}

__global__ void unfreezeAllClusters(SimulationData data)
{
    auto const clusterPartition = calcPartition(
        data.entities.clusterFreezedPointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto const& clusterFreezed = data.entities.clusterFreezedPointers.at(clusterIndex);
        if (clusterFreezed && nullptr == clusterFreezed->clusterToFuse) {
            auto clusterPointer = data.entities.clusterPointers.getNewElement();
            *clusterPointer = clusterFreezed;
            clusterFreezed->setUnfreezed();
        }
    }
}

__global__ void cleanupCellMapFreezed(SimulationData data)
{
    data.cellMap.cleanupFreezed_gridCall();
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void calcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.dynamicMemory.reset();
    KERNEL_CALL(resetCellFunctionData, data);
    KERNEL_CALL(clusterProcessingStep1, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep1, data);
    KERNEL_CALL(tokenProcessingStep2, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep3, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep2, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep3, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep4, data, data.entities.clusterPointers.getNumEntries());

    KERNEL_CALL(particleProcessingStep1, data);
    KERNEL_CALL(particleProcessingStep2, data);
    KERNEL_CALL(particleProcessingStep3, data);

    KERNEL_CALL(freezeClustersIfAllowed, data);
    if ((data.timestep % 5) == 0) {
        KERNEL_CALL(cleanupCellMapFreezed, data);
        data.cellMap.resetFreezed();

        KERNEL_CALL(unfreezeAllClusters, data);
        data.entities.clusterFreezedPointers.reset();
    }

    cleanup << <1, 1 >> > (data);
    
    cudaDeviceSynchronize();

}

