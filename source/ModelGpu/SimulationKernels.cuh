#pragma once

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

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterProcessingOnOrigDataStep1_blockCall(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingMovement_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep2_blockCall(SimulationData &data, int clusterIndex)
{
    ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.destroyCloseCell_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep3_blockCall(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingRadiation_blockCall();
    clusterProcessor.processingCollision_blockCall();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingOnCopyData_blockCall(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnCopyData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingDecomposition_blockCall();
    clusterProcessor.processingClusterCopy_blockCall();
}

__global__ void clusterProcessingOnOrigDataStep1(SimulationData data)
{
	BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
	for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep1_blockCall(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnOrigDataStep2(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep2_blockCall(data, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep3(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep3_blockCall(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnCopyData(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
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
    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
    for (int index = clusterBlock.startIndex; index <= clusterBlock.endIndex; ++index) {
        tokenProcessingStep1_blockCall(data, index);
    }
}

__global__ void tokenProcessingStep2(SimulationData data)
{
    BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);
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

__device__ void clearCellCluster(SimulationData const &data, int clusterIndex)
{
	auto const &oldCluster = data.clusters.getEntireArray()[clusterIndex];

	int2 size = data.size;
	int oldNumCells = oldCluster.numCellPointers;
	Map<Cell> map;
	map.init(size, data.cellMap);

	BlockData cellBlock = calcPartition(oldNumCells, threadIdx.x, blockDim.x);
	for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
		float2 absPos = oldCluster.cellPointers[cellIndex]->absPos;
		map.set(absPos, nullptr);
	}
}

__device__ void clearParticle(SimulationData const &data, int particleIndex)
{
	Map<Particle> map;
	map.init(data.size, data.particleMap);

	auto const &particle = data.particles.getEntireArray()[particleIndex];
	map.set(particle.pos, nullptr);
}

__global__ void clearMaps(SimulationData data)
{
	BlockData clusterBlock = calcPartition(data.clusters.getNumEntries(), blockIdx.x, gridDim.x);

	for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
		clearCellCluster(data, clusterIndex);
	}

	BlockData particleBlock = 
        calcPartition(data.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
	for (int particleIndex = particleBlock.startIndex; particleIndex <= particleBlock.endIndex; ++particleIndex) {
		clearParticle(data, particleIndex);
	}
}
