#pragma once

#include <cooperative_groups.h>
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterProcessorOnOrigData.cuh"
#include "ClusterProcessorOnCopyData.cuh"
#include "ParticleProcessorOnOrigData.cuh"
#include "ParticleProcessorOnCopyData.cuh"
#include "TokenProcessor.cuh"
#include "CleanupKernels.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterProcessingOnOrigDataStep1_blockCall(SimulationData data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingMovement_blockCall();
}

__device__  void clusterProcessingOnOrigDataStep2_blockCall(SimulationData data, int clusterIndex)
{
    ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.destroyCloseCell_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep3_blockCall(SimulationData data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingRadiation_blockCall();
    clusterProcessor.processingCollision_blockCall();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingOnCopyData_blockCall(SimulationData data, int clusterIndex)
{
	ClusterProcessorOnCopyData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingDecomposition_blockCall();
    clusterProcessor.processingClusterCopy_blockCall();
}

__device__ int getNumClusterPointers(SimulationData const& data)
{
    int numClusterPointers = data.clusterPointers.getNumEntries();
//    cooperative_groups::this_grid().sync();

    return numClusterPointers;
}

__global__ void clusterProcessingOnOrigDataStep1(SimulationData data)
{
/*
    int* clusterIndexPtr = new int;
    int& clusterIndex = *clusterIndexPtr;

    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        auto numCells = data.clusters.at(clusterIndex)->numCellPointers;
        clusterProcessingOnOrigDataStep1_blockCall << <1, min(64, numCells) >> > (data, clusterIndex);
    }

    cudaDeviceSynchronize();
    delete clusterIndexPtr;
*/
    int numClusterPointers = getNumClusterPointers(data);

    BlockData clusterBlock = calcPartition(numClusterPointers, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep1_blockCall(data, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep2(SimulationData data)
{
    int numClusterPointers = getNumClusterPointers(data);

    BlockData clusterBlock = calcPartition(numClusterPointers, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep2_blockCall(data, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep3(SimulationData data)
{
    int numClusterPointers = getNumClusterPointers(data);

    BlockData clusterBlock = calcPartition(numClusterPointers, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep3_blockCall(data, clusterIndex);
    }
}

__global__ void clusterProcessingOnCopyData(SimulationData data)
{
    int numClusterPointers = getNumClusterPointers(data);

    BlockData clusterBlock = calcPartition(numClusterPointers, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnCopyData_blockCall(data, clusterIndex);
    }
}


/************************************************************************/
/* Tokens																*/
/************************************************************************/

__device__ void tokenProcessingStep1_blockCall(SimulationData data, int clusterIndex)
{
	TokenProcessor tokenProcessor;
    tokenProcessor.init_blockCall(data, clusterIndex);
    tokenProcessor.processingEnergyAveraging_blockCall();
}

__device__ void tokenProcessingStep2_blockCall(SimulationData data, int clusterIndex)
{
	TokenProcessor tokenProcessor;

    tokenProcessor.init_blockCall(data, clusterIndex);
    tokenProcessor.processingSpreading_blockCall();
    tokenProcessor.processingFeatures_blockCall();
}

__global__ void tokenProcessingStep1(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int index = clusterBlock.startIndex; index <= clusterBlock.endIndex; ++index) {
        tokenProcessingStep1_blockCall(data, index);
    }
}

__global__ void tokenProcessingStep2(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusterPointers.getNumEntries(), blockIdx.x, gridDim.x);
    for (int index = clusterBlock.startIndex; index <= clusterBlock.endIndex; ++index) {
		tokenProcessingStep2_blockCall(data, index);
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
    data.particlesNew.reset();

    tokenProcessingStep1<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(data);
    cudaDeviceSynchronize();
    tokenProcessingStep2 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    clusterProcessingOnOrigDataStep1 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    clusterProcessingOnOrigDataStep2 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    clusterProcessingOnOrigDataStep3 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    clusterProcessingOnCopyData << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingOnOrigDataStep1 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingOnOrigDataStep2 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();
    particleProcessingOnCopyData << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK >> > (data);
    cudaDeviceSynchronize();

    cleanup<<<1, 1>>>(data);
    cudaDeviceSynchronize();
}
