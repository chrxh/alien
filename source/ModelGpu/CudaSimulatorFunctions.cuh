#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ClusterDynamics.cuh"
#include "ClusterBuilder.cuh"
#include "ParticleDynamics.cuh"
#include "ParticleBuilder.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterBuilding(SimulationDataInternal &data, int clusterIndex)
{
	__shared__ ClusterBuilder reassembler;
	if (0 == threadIdx.x) {
		reassembler.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(reassembler.getNumOrigCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	reassembler.processingDecomposition(startCellIndex, endCellIndex);
	reassembler.processingDataCopy(startCellIndex, endCellIndex);
}

__global__ void clusterBuilding(SimulationDataInternal data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterBuilding(data, clusterIndex);
	}
}

__device__ void clusterDynamicsStep1(SimulationDataInternal &data, int clusterIndex)
{
	__shared__ ClusterDynamics mover;
	if (0 == threadIdx.x) {
		mover.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(mover.getNumCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	mover.processingMovement(startCellIndex, endCellIndex);
}

__global__ void clusterDynamicsStep1(SimulationDataInternal data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterDynamicsStep1(data, clusterIndex);
	}
}

__device__ void clusterDynamicsStep2(SimulationDataInternal &data, int clusterIndex)
{
	__shared__ ClusterDynamics mover;
	if (0 == threadIdx.x) {
		mover.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(mover.getNumCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	mover.destroyCloseCell(startCellIndex, endCellIndex);
	mover.processingRadiation(startCellIndex, endCellIndex);
	mover.processingCollision(startCellIndex, endCellIndex);	//attention: can result an temporarily inconsistent state
}

__global__ void clusterDynamicsStep2(SimulationDataInternal data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterDynamicsStep2(data, clusterIndex);
	}
}

/************************************************************************/
/* Particles															*/
/************************************************************************/

__global__ void particleDynamicsStep1(SimulationDataInternal data)
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
}

__global__ void particleDynamicsStep2(SimulationDataInternal data)
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

__global__ void particleBuilding(SimulationDataInternal data)
{
	__shared__ ParticleBuilder blockProcessor;
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

__device__ void clearCellCluster(SimulationDataInternal const &data, int clusterIndex)
{
	auto const &oldCluster = data.clustersAC1.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster.numCells;
	if (threadIdx.x >= oldNumCells) {
		return;
	}
	Map<CellData> map;
	map.init(size, data.cellMap);

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster.cells[cellIndex].absPos;
		map.set(absPos, nullptr);
	}
}

__device__ void clearParticle(SimulationDataInternal const &data, int particleIndex)
{
	Map<ParticleData> map;
	map.init(data.size, data.particleMap);

	auto const &particle = data.particlesAC1.getEntireArray()[particleIndex];
	map.set(particle.pos, nullptr);
}

__global__ void clearMaps(SimulationDataInternal data)
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
