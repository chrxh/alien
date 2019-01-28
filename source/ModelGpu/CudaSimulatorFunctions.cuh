#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "BlockProcessorForCluster.cuh"
#include "BlockProcessorForParticles.cuh"

/************************************************************************/
/* Clusters																*/
/************************************************************************/

__device__ void clusterMovement(SimulationDataInternal &data, int clusterIndex)
{
	__shared__ BlockProcessorForCluster1 blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(blockProcessor.getNumOrigCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	blockProcessor.processingMovementAndCollision(startCellIndex, endCellIndex);
	blockProcessor.processingRadiation(startCellIndex, endCellIndex);
	blockProcessor.processingDecomposition(startCellIndex, endCellIndex);
	blockProcessor.processingDataCopy(startCellIndex, endCellIndex);
}

__global__ void clusterMovement(SimulationDataInternal data)
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
		clusterMovement(data, clusterIndex);
	}
}

__device__ void clusterCollision(SimulationDataInternal &data, int clusterIndex)
{
	__shared__ BlockProcessorForCluster2 blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data, clusterIndex);
	}
	__syncthreads();

	int startCellIndex;
	int endCellIndex;
	calcPartition(blockProcessor.getNumCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	blockProcessor.processingCollision(startCellIndex, endCellIndex);
}

__global__ void clusterCollision(SimulationDataInternal data)
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
		clusterCollision(data, clusterIndex);
	}
}

/************************************************************************/
/* Particles															*/
/************************************************************************/

__global__ void particleCollision(SimulationDataInternal data)
{
	__shared__ BlockProcessorForParticles blockProcessor;
	if (0 == threadIdx.x) {
		blockProcessor.init(data);
	}
	__syncthreads();

	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC2.getNumEntries();

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	blockProcessor.processingCollision(startIndex, endIndex);
}

__global__ void particleMovement(SimulationDataInternal data)
{
	__shared__ BlockProcessorForParticles blockProcessor;
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
	map.init(size, data.cellMap1, data.cellMap2);

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster.cells[cellIndex].absPos;
		map.setToOrigMap(absPos, nullptr);
	}
}

__device__ void clearParticle(SimulationDataInternal const &data, int particleIndex)
{
	Map<ParticleData> map;
	map.init(data.size, data.particleMap1, data.particleMap2);

	auto const &particle = data.particlesAC1.getEntireArray()[particleIndex];
	map.setToOrigMap(particle.pos, nullptr);
}

__global__ void clearMaps(SimulationDataInternal data)
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
		clearCellCluster(data, clusterIndex);
	}

	indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	numEntities = data.particlesAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		clearParticle(data, particleIndex);
	}

}
