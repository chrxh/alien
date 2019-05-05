#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterProcessorOnOrigData.cuh"
#include "ClusterProcessorOnCopyData.cuh"
#include "ParticleProcessorOnOrigData.cuh"
#include "ParticleProcessorOnCopyData.cuh"
#include "TokenProcessorOnOrigData.cuh"
#include "TokenProcessorOnCopyData.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterProcessingOnOrigDataStep1(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData dynamics;
	dynamics.init(data, clusterIndex);
	dynamics.processingMovement();
}

__device__ void clusterProcessingOnOrigDataStep2(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnOrigData dynamics;
	dynamics.init(data, clusterIndex);
	dynamics.destroyCloseCell();
	dynamics.processingRadiation();
	dynamics.processingCollision();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterProcessingOnCopyData(SimulationData &data, int clusterIndex)
{
	ClusterProcessorOnCopyData reorganizer;
	reorganizer.init(data, clusterIndex);
	reorganizer.processingDecomposition();
	reorganizer.processingClusterCopy();
}

__global__ void clusterProcessingOnOrigDataStep1(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep1(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnOrigDataStep2(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterProcessingOnOrigDataStep2(data, clusterIndex);
	}
}

__global__ void clusterProcessingOnCopyData(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

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

__global__ void tokenProcessingOnOrigData(SimulationData data)
{
	TokenProcessorOnOrigData dynamics;
	dynamics.init(data);
	dynamics.processingEnergyGuidance();
}

__device__ void tokenProcessingOnCopyData(SimulationData data, int clusterIndex)
{
	TokenProcessorOnCopyData reorganizer;

	reorganizer.init(data, clusterIndex);
	reorganizer.processingTokenSpreading();
}

__global__ void tokenProcessingOnCopyData(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

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
	ParticleProcessorOnOrigData dynamics;
	dynamics.init(data);
	dynamics.processingMovement();
	dynamics.processingTransformation();
}

__global__ void particleProcessingOnOrigDataStep2(SimulationData data)
{
	ParticleProcessorOnOrigData dynamics;
	dynamics.init(data);
	dynamics.processingCollision();
}

__global__ void particleReorganizing(SimulationData data)
{
	ParticleProcessorOnCopyData reorganizer;
	reorganizer.init(data);
	reorganizer.processingDataCopy();
}

__device__ void clearCellCluster(SimulationData const &data, int clusterIndex)
{
	auto const &oldCluster = data.clustersAC1.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster.numCells;
	if (threadIdx.x >= oldNumCells) {
		return;
	}
	Map<Cell> map;
	map.init(size, data.cellMap);

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster.cells[cellIndex].absPos;
		map.set(absPos, nullptr);
	}
}

__device__ void clearParticle(SimulationData const &data, int particleIndex)
{
	Map<Particle> map;
	map.init(data.size, data.particleMap);

	auto const &particle = data.particlesAC1.getEntireArray()[particleIndex];
	map.set(particle.pos, nullptr);
}

__global__ void clearMaps(SimulationData data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);

	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clearCellCluster(data, clusterIndex);
	}

	indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	numEntities = data.particlesAC1.getNumEntries();
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		clearParticle(data, particleIndex);
	}
}
