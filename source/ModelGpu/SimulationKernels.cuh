#pragma once

#include <cooperative_groups.h>
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "TokenProcessor.cuh"
#include "CleanupKernels.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__global__ void clusterProcessingStep1(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
        clusterProcessor.processingMovement_blockCall();
        clusterProcessor.updateMap_blockCall();
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
        clusterProcessor.destroyCell_blockCall();
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
        clusterProcessor
            .processingCollision_blockCall();  //attention: can result a temporarily inconsistent state, will be resolved in step 4
        clusterProcessor.processingRadiation_blockCall();
    }
}

__global__ void clusterProcessingStep4(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
        clusterProcessor.processingCellDeath_blockCall();
        clusterProcessor.processingMutation_blockCall();
        clusterProcessor.processingDecomposition_blockCall();
        clusterProcessor.processingClusterCopy_blockCall();
    }
}


/************************************************************************/
/* Tokens																*/
/************************************************************************/

__global__ void tokenProcessingStep1(SimulationData data, int clusterArrayIndex)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.init_gridCall(data, clusterArrayIndex);
    tokenProcessor.processingEnergyAveraging_gridCall();
    tokenProcessor.processingSpreading_gridCall();
    tokenProcessor.processingLightWeigthedFeatures_gridCall();
}

__global__ void tokenProcessingStep2(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
        tokenProcessor.processingHeavyWeightedFeatures_blockCall();
    }
}

/************************************************************************/
/* Particles															*/
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
/* Debug      															*/
/************************************************************************/
__global__ void DEBUG_checkCluster(SimulationData data, int numClusters, int parameter)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto const clusterPointer = &data.entities.clusterPointerArrays.getArray(0).at(clusterIndex);
        DEBUG_cluster::check_blockCall(&data, *clusterPointer, parameter);
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void calcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.arrays.reset();

     printf("STEP 1\n");
    MULTI_CALL(clusterProcessingStep1, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
     printf("STEP 2\n");
    MULTI_CALL(tokenProcessingStep1, data);
     printf("STEP 3\n");
    MULTI_CALL(tokenProcessingStep2, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
     printf("STEP 4\n");
    MULTI_CALL(clusterProcessingStep2, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
     printf("STEP 5\n");
    MULTI_CALL(clusterProcessingStep3, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
     printf("STEP 6\n");
    MULTI_CALL(clusterProcessingStep4, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
     printf("STEP 7\n");
    particleProcessingStep1 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
     printf("STEP 8\n");
    particleProcessingStep2 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
     printf("STEP 9\n");
    particleProcessingStep3 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
     printf("STEP 10\n");

    cleanup << <1, 1 >> > (data);
    cudaDeviceSynchronize();
}

