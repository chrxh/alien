#pragma once

#include <cooperative_groups.h>
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterProcessor.cuh"
#include "ParticleProcessorOnOrigData.cuh"
#include "ParticleProcessorOnCopyData.cuh"
#include "TokenProcessor.cuh"
#include "CleanupKernels.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterProcessingOnOrigDataStep1_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingMovement_blockCall();
}

__device__  void clusterProcessingOnOrigDataStep2_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
    ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.destroyCloseCell_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep3_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
    ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingRadiation_blockCall();
    clusterProcessor.processingCollision_blockCall();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingOnCopyData_blockCall(SimulationData data, int clusterArrayIndex, int clusterIndex)
{
	ClusterProcessor clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterArrayIndex, clusterIndex);
    clusterProcessor.processingDecomposition_blockCall();
    clusterProcessor.processingClusterCopy_blockCall();
}

__global__ void clusterProcessingOnOrigDataStep1(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep1_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep2(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep2_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep3(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep3_blockCall(data, clusterArrayIndex, clusterIndex);
    }
}

__global__ void clusterProcessingOnCopyData(SimulationData data, int numClusters, int clusterArrayIndex)
{
    BlockData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnCopyData_blockCall(data, clusterArrayIndex, clusterIndex);
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

__global__ void particleProcessingOnOrigDataStep1(SimulationData data)
{
	ParticleProcessorOnOrigData particleProcessor;
    particleProcessor.init_blockCall(data);
    particleProcessor.processingMovement_blockCall();
    particleProcessor.processingTransformation_blockCall();
}

__global__ void particleProcessingOnOrigDataStep2(SimulationData data)
{
	ParticleProcessorOnOrigData particleProcessor;
    particleProcessor.init_blockCall(data);
    particleProcessor.processingCollision_blockCall();
}

__global__ void particleProcessingOnCopyData(SimulationData data)
{
	ParticleProcessorOnCopyData particleProcessor;
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
    MULTI_CALL(clusterProcessingOnOrigDataStep1, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingOnOrigDataStep2, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingOnOrigDataStep3, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    MULTI_CALL(clusterProcessingOnCopyData, data, data.entities.clusterPointerArrays.getArray(i).getNumEntries());
    particleProcessingOnOrigDataStep1 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingOnOrigDataStep2 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingOnCopyData << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();

    cleanup<<<1, 1>>>(data);
    cudaDeviceSynchronize();
}

