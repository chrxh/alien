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
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep1_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep2_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep3_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingStep4(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingStep4_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}


/************************************************************************/
/* Tokens																*/
/************************************************************************/

__device__ void tokenProcessingStep1_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	TokenProcessor tokenProcessor;
    tokenProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    tokenProcessor.processingEnergyAveraging_blockCall();
}

__device__ void tokenProcessingStep2_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	TokenProcessor tokenProcessor;

    tokenProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    tokenProcessor.processingSpreading_blockCall();
    tokenProcessor.processingFeatures_blockCall();
}

__global__ void tokenProcessingStep1(SimulationData data, int clusterArrayIndex)
{
    auto const& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        tokenProcessingStep1_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void tokenProcessingStep2(SimulationData data, int clusterArrayIndex)
{
    auto const& clusters = data.entities.clusterPointerArrays.getArray(clusterArrayIndex);
    BlockData clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        tokenProcessingStep2_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}


/************************************************************************/
/* Particles															*/
/************************************************************************/

__global__ void particleProcessingStep1(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_blockCall(data);
    particleProcessor.processingMovement_blockCall();
    particleProcessor.processingTransformation_blockCall();
}

__global__ void particleProcessingStep2(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.init_blockCall(data);
    particleProcessor.processingCollision_blockCall();
}

__global__ void particleProcessingStep3(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_blockCall(data);
    particleProcessor.processingDataCopy_blockCall();
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void calcSimulationTimestep(SimulationData data)
{
    MULTI_CALL(tokenProcessingStep1, data);
    MULTI_CALL(tokenProcessingStep2, data);
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

