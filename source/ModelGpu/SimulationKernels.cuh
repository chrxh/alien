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

__device__ void clusterProcessingStep1_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingMovement_blockCall();
    clusterProcessor.updateMap_blockCall();
}

__device__  void clusterProcessingStep2_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
    ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.destroyCloseCell_blockCall();
}

__device__ void clusterProcessingStep3_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
    ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingRadiation_blockCall();
    clusterProcessor.processingCollision_blockCall();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingStep4_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingDecomposition_blockCall();
    clusterProcessor.processingClusterCopy_blockCall();
}

__global__ void clusterProcessingStep1(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep1_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep2_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep3_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep4(SimulationData data, int numClusters, int clusterArrayIndex)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep4_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}


/************************************************************************/
/* Tokens																*/
/************************************************************************/

__global__ void tokenProcessing(SimulationData data, int clusterArrayIndex)
{
    TokenProcessor tokenProcessor;
    tokenProcessor.init_gridCall(data, clusterArrayIndex);
    tokenProcessor.processingEnergyAveraging_gridCall();
    tokenProcessor.processingSpreading_gridCall();
    tokenProcessor.processingFeatures_gridCall();
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
/* Main      															*/
/************************************************************************/

__global__ void calcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();

    MULTI_CALL(tokenProcessing, data);
    MULTI_CALL(clusterProcessingStep1, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingStep2, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingStep3, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingStep4, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    particleProcessingStep1 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingStep2 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingStep3 << <cudaConstants.NUM_BLOCKS, cudaConstants.NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();

    cleanup<<<1, 1>>>(data);
    cudaDeviceSynchronize();
}

