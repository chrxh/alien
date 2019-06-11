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

__device__ void clusterProcessingOnOrigDataStep1(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingMovement_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep2(SimulationData &data, int clusterIndex)
{
    ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.destroyCloseCell_blockCall();
}

__device__ void clusterProcessingOnOrigDataStep3(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingRadiation_blockCall();
    clusterProcessor.processingCollision_blockCall();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingOnCopyData(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnCopyData clusterProcessor;
    clusterProcessor.init_blockCall(data, clusterIndex);
    clusterProcessor.processingDecomposition_blockCall();
    clusterProcessor.processingClusterCopy_blockCall();
}

__global__ void clusterProcessingOnOrigDataStep1(SimulationData data)
{
	int numEntities = data.clusters.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep1(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnOrigDataStep2(SimulationData data)
{
    int numEntities = data.clusters.getNumEntries();

    int startIndex;
    int endIndex;
    calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
    for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
        clusterProcessingOnOrigDataStep2(data, clusterIndex);
    }
}

__global__ void clusterProcessingOnOrigDataStep3(SimulationData data)
{
	int numEntities = data.clusters.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep3(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnCopyData(SimulationData data)
{
	int numEntities = data.clusters.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterProcessingOnCopyData(data, clusterIndex);
	}
}


/************************************************************************/
/* Tokens																*/
/************************************************************************/

__device__ void tokenProcessingOnOrigData(SimulationData data, int clusterIndex)
{
	TokenProcessor tokenProcessor;
    tokenProcessor.init(data, clusterIndex);
    tokenProcessor.processingEnergyAveraging();
}

__device__ void tokenProcessingOnCopyData(SimulationData data, int clusterIndex)
{
	TokenProcessor tokenProcessor;

    tokenProcessor.init(data, clusterIndex);
    tokenProcessor.processingSpreadingAndFeatures_blockCall();
}

__global__ void tokenProcessingOnOrigData(SimulationData data)
{
    int numEntities = data.clusters.getNumEntries();

    int startIndex;
    int endIndex;
    calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
    for (int index = startIndex; index <= endIndex; ++index) {
        tokenProcessingOnOrigData(data, index);
    }
}

__global__ void tokenProcessingOnCopyData(SimulationData data)
{
	int numEntities = data.clusters.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int index = startIndex; index <= endIndex; ++index) {
		tokenProcessingOnCopyData(data, index);
	}
}


/************************************************************************/
/* Particles															*/
/************************************************************************/

__global__ void particleProcessingOnOrigDataStep1(SimulationData data)
{
	ParticleProcessorOnOrigData particleProcessor;
    particleProcessor.init(data);
    particleProcessor.processingMovement();
    particleProcessor.processingTransformation();
}

__global__ void particleProcessingOnOrigDataStep2(SimulationData data)
{
	ParticleProcessorOnOrigData particleProcessor;
    particleProcessor.init(data);
    particleProcessor.processingCollision();
}

__global__ void particleProcessingOnCopyData(SimulationData data)
{
	ParticleProcessorOnCopyData particleProcessor;
    particleProcessor.init(data);
    particleProcessor.processingDataCopy();
}

__device__ void clearCellCluster(SimulationData const &data, int clusterIndex)
{
	auto const &oldCluster = data.clusters.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster.numCellPointers;
	Map<Cell> map;
	map.init(size, data.cellMap);

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
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
	int indexResource = blockIdx.x;
	int numEntities = data.clusters.getNumEntries();
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clearCellCluster(data, clusterIndex);
	}

	indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	numEntities = data.particles.getNumEntries();
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		clearParticle(data, particleIndex);
	}
}
