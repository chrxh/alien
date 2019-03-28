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
	__shared__ ClusterDynamics cluster;
	if (0 == threadIdx.x) {
		cluster.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(cluster.getNumCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	cluster.processingMovement(startCellIndex, endCellIndex);
}

__device__ void clusterDynamicsStep2(SimulationData &data, int clusterIndex)
{
	__shared__ ClusterDynamics cluster;
	if (0 == threadIdx.x) {
		cluster.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(cluster.getNumCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	cluster.destroyCloseCell(startCellIndex, endCellIndex);
	cluster.processingRadiation(startCellIndex, endCellIndex);
	cluster.processingCollision(startCellIndex, endCellIndex);	//attention: can result a temporarily inconsistent state
																//will be resolved in reorganizer
}

__device__ void clusterReorganizing(SimulationData &data, int clusterIndex)
{
	__shared__ ClusterReorganizer builder;
	if (0 == threadIdx.x) {
		builder.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(builder.getNumOrigCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	builder.processingDecomposition(startCellIndex, endCellIndex);
	builder.processingDataCopy(startCellIndex, endCellIndex);
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
	__shared__ ParticleDynamics blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data);
	}
	__syncthreads();

	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);

	blockProcessor.processingMovement(startIndex, endIndex);
	blockProcessor.processingTransformation(startIndex, endIndex);
}

__global__ void particleDynamicsStep2(SimulationData data)
{
	__shared__ ParticleDynamics blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data);
	}
	__syncthreads();

	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);

	blockProcessor.processingCollision(startIndex, endIndex);
}

__global__ void particleReorganizing(SimulationData data)
{
	__shared__ ParticleReorganizer blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data);
	}
	__syncthreads();

	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC1.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);

	blockProcessor.processingDataCopy(startIndex, endIndex);
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
