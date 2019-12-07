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
        clusterProcessor.init_blockCall(data, 0, clusterIndex);
        clusterProcessor.processingMovement_blockCall();
        clusterProcessor.updateMap_blockCall();
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, 0, clusterIndex);
        clusterProcessor.destroyCell_blockCall();
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, 0, clusterIndex);
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
        clusterProcessor.init_blockCall(data, 0, clusterIndex);
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
    tokenProcessor.init_gridCall(data, 0);
    tokenProcessor.processingEnergyAveraging_gridCall();
    tokenProcessor.processingSpreading_gridCall();
    tokenProcessor.processingLightWeigthedFeatures_gridCall();
}

__global__ void tokenProcessingStep2(SimulationData data, int numClusters)
{
    auto const clusterPartition = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_blockCall(data, 0, clusterIndex);
        tokenProcessor.createCellFunctionData_blockCall();
    }
}

__global__ void tokenProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_blockCall(data, 0, clusterIndex);
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
/* Main      															*/
/************************************************************************/

__global__ void calcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.dynamicMemory.reset();
    KERNEL_CALL(resetCellFunctionData, data);
    KERNEL_CALL(clusterProcessingStep1, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());
    KERNEL_CALL(tokenProcessingStep1, data);
    KERNEL_CALL(tokenProcessingStep2, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());
    KERNEL_CALL(tokenProcessingStep3, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());
    KERNEL_CALL(clusterProcessingStep2, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());
    KERNEL_CALL(clusterProcessingStep3, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());
    KERNEL_CALL(clusterProcessingStep4, data, data.entities.clusterPointerArrays.getArray(0).getNumEntries());

    KERNEL_CALL(particleProcessingStep1, data);
    KERNEL_CALL(particleProcessingStep2, data);
    KERNEL_CALL(particleProcessingStep3, data);

    cleanup << <1, 1 >> > (data);
    cudaDeviceSynchronize();

}

