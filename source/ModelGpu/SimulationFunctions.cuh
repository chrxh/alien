#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterDynamics.cuh"
#include "ClusterReorganizer.cuh"
#include "ParticleDynamics.cuh"
#include "ParticleReorganizer.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterDynamicsStep1(SimulationData &data, int clusterIndex)
{
	ClusterDynamics cluster;
	cluster.init(data, clusterIndex);

	cluster.processingMovement();
}

__device__ void clusterDynamicsStep2(SimulationData &data, int clusterIndex)
{
	ClusterDynamics cluster;
	cluster.init(data, clusterIndex);

	cluster.destroyCloseCell();
	cluster.processingRadiation();
	cluster.processingCollision();	//attention: can result a temporarily inconsistent state
									//will be resolved in reorganizer
}

__device__ void clusterReorganizing(SimulationData &data, int clusterIndex)
{
	ClusterReorganizer reorganizer;
	reorganizer.init(data, clusterIndex);
	
	reorganizer.processingDecomposition();
	reorganizer.processingClusterCopy();
	reorganizer.processingTokenCopy();
}

__global__ void clusterDynamicsStep1(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterDynamicsStep1(data, clusterIndex);
	}
}

__global__ void clusterDynamicsStep2(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterDynamicsStep2(data, clusterIndex);
	}
}

__global__ void clusterReorganizing(SimulationData data)
{
	int numEntities = data.clustersAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, blockIdx.x, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterReorganizing(data, clusterIndex);
	}
}


/************************************************************************/
/* Particles															*/
/************************************************************************/

__global__ void particleDynamicsStep1(SimulationData data)
{
	ParticleDynamics particle;
	particle.init(data);
	particle.processingMovement();
	particle.processingTransformation();
}

__global__ void particleDynamicsStep2(SimulationData data)
{
	ParticleDynamics blockProcessor;
	blockProcessor.init(data);
	blockProcessor.processingCollision();
}

__global__ void particleReorganizing(SimulationData data)
{
	ParticleReorganizer blockProcessor;
	blockProcessor.init(data);
	blockProcessor.processingDataCopy();
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
