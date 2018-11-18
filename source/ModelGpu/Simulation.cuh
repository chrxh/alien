#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "Constants.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "BlockProcessorForCluster.cuh"

__device__ void clusterMovement(SimulationData &data, int clusterIndex)
{
	__shared__ BlockProcessorForCluster blockProcessor;
	blockProcessor.init(data, clusterIndex);

	int startCellIndex;
	int endCellIndex;
	calcPartition(blockProcessor.getNumOrigCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	blockProcessor.processingCollision(startCellIndex, endCellIndex);
	blockProcessor.processingMovement(startCellIndex, endCellIndex);
	blockProcessor.processingRadiation(startCellIndex, endCellIndex);
}

__global__ void clusterMovement(SimulationData data)
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

__device__ void particleMovement(SimulationData &data, int particleIndex)
{
	ParticleData *oldParticle = &data.particlesAC1.getEntireArray()[particleIndex];
	__shared__ Map<CellData> map;
	map.init(data.size, data.cellMap1, data.cellMap2);
	auto cell = map.getFromOrigMap(toInt2(oldParticle->pos));
	if (cell) {
		auto nextCell = cell->nextTimestep;
		atomicAdd(&nextCell->energy, oldParticle->energy);
		return;
	}
	ParticleData *newParticle = data.particlesAC2.getNewElement();
	*newParticle = *oldParticle;
	newParticle->pos = add(newParticle->pos, newParticle->vel);
	map.mapPosCorrection(newParticle->pos);
}

__global__ void particleMovement(SimulationData data)
{
	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		particleMovement(data, particleIndex);
	}
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
	Map<CellData> map;
	map.init(size, data.cellMap1, data.cellMap2);

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster.cells[cellIndex].absPos;
		map.setToOrigMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, nullptr);
	}
}

__device__ void clearParticle(SimulationData const &data, int particleIndex)
{
	Map<ParticleData> map;
	map.init(data.size, data.particleMap1, data.particleMap2);

	auto const &particle = data.particlesAC1.getEntireArray()[particleIndex];
	map.setToOrigMap(toInt2(particle.pos), nullptr);
}

__global__ void clearMaps(SimulationData data)
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
