#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "Constants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

struct BlockProcessorForCluster
{
	__inline__ __device__ void init(SimulationData& data, int clusterIndex)
	{
		if (0 == threadIdx.x ) {
			_data = &data;
			origCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
			clusterCopy = *origCluster;
			newCells = _data->cellsAC2.getNewSubarray(clusterCopy.numCells);
			newCluster = _data->clustersAC2.getNewElement();
			calcRotationMatrix(clusterCopy.angle, rotMatrix);
			atLeastOneCellDestroyed = false;
			collisionData.init();
		}
		__syncthreads();
	}

	__inline__ __device__ int getNumOrigCells()
	{
		return origCluster->numCells;
	}

	__inline__ __device__ void processingCollision(int startCellIndex, int endCellIndex)
	{
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData *origCell = &origCluster->cells[cellIndex];
			collectCollisionDataForCell(*_data, origCell, collisionData);
			if (!origCell->alive) {
				atLeastOneCellDestroyed = true;
			}
		}

		__syncthreads();

		if (0 == threadIdx.x) {

			for (int i = 0; i < collisionData.numEntries; ++i) {
				CollisionEntry* entry = &collisionData.entries[i];
				float numCollisions = static_cast<float>(entry->numCollisions);
				entry->collisionPos.x /= numCollisions;
				entry->collisionPos.y /= numCollisions;
				entry->normalVec.x /= numCollisions;
				entry->normalVec.y /= numCollisions;
				calcCollision(&clusterCopy, entry, _data->size);
			}

			clusterCopy.angle += clusterCopy.angularVel;
			angleCorrection(clusterCopy.angle);
			clusterCopy.pos = add(clusterCopy.pos, clusterCopy.vel);
			mapPosCorrection(clusterCopy.pos, _data->size);
			clusterCopy.numCells = 0;
			clusterCopy.cells = newCells;
		}
		__syncthreads();
	}

	__inline__ __device__ void processingMovement(int startCellIndex, int endCellIndex)
	{
		int numOrigCells = origCluster->numCells;
		if (threadIdx.x < numOrigCells) {
			for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
				copyAndMoveCell(cellIndex);
			}
		}
		__syncthreads();
	}

	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex)
	{
		int numOrigCells = origCluster->numCells;
		if (threadIdx.x < numOrigCells) {
			for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
				CellData *origCell = &origCluster->cells[cellIndex];
				cellRadiation(origCell);
			}
		}
		__syncthreads();
	}

	ClusterData clusterCopy;
	ClusterData *origCluster;
	ClusterData *newCluster;
	CellData *newCells;
	float rotMatrix[2][2];
	CollisionData collisionData;
	bool atLeastOneCellDestroyed;

private:
	__inline__ __device__  void copyAndMoveCell(int cellIndex)
	{
		CellData *oldCell = &origCluster->cells[cellIndex];
		CellData cellCopy = *oldCell;

		float2 absPos;
		absPos.x = cellCopy.relPos.x*rotMatrix[0][0] + cellCopy.relPos.y*rotMatrix[0][1] + clusterCopy.pos.x;
		absPos.y = cellCopy.relPos.x*rotMatrix[1][0] + cellCopy.relPos.y*rotMatrix[1][1] + clusterCopy.pos.y;
		cellCopy.absPos = absPos;
		cellCopy.cluster = newCluster;
		if (cellCopy.protectionCounter > 0) {
			--cellCopy.protectionCounter;
		}
		if (cellCopy.setProtectionCounterForNextTimestep) {
			cellCopy.protectionCounter = PROTECTION_TIMESTEPS;
			cellCopy.setProtectionCounterForNextTimestep = false;
		}
		int newCellIndex = atomicAdd(&clusterCopy.numCells, 1);
		CellData *newCell = &newCells[newCellIndex];
		*newCell = cellCopy;
		setToMap<CellData>({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell, _data->cellMap2, _data->size);

		oldCell->nextTimestep = newCell;
	}

	__inline__ __device__ void cellRadiation(CellData *cell)
	{
		if (_data->numberGen.random() < RADIATION_PROB) {
			createNewParticle(cell);
		}
		if (cell->energy < CELL_MIN_ENERGY) {
			cell->alive = false;
		}
	}

	__inline__ __device__ void createNewParticle(CellData *cell)
	{
		auto particle = _data->particlesAC2.getNewElement();
		auto &pos = cell->absPos;
		particle->id = _data->numberGen.newId_Kernel();
		particle->pos = { pos.x + _data->numberGen.random(2.0f) - 1.0f, pos.y + _data->numberGen.random(2.0f) - 1.0f };
		mapPosCorrection(particle->pos, _data->size);
		particle->vel = { (_data->numberGen.random() - 0.5f) * RADIATION_VELOCITY_PERTURBATION
			, (_data->numberGen.random() - 0.5f) * RADIATION_VELOCITY_PERTURBATION };
		float radiationEnergy = powf(cell->energy, RADIATION_EXPONENT) * RADIATION_FACTOR;
		radiationEnergy = radiationEnergy / RADIATION_PROB;
		radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
		if (radiationEnergy > cell->energy - 1) {
			radiationEnergy = cell->energy - 1;
		}
		particle->energy = radiationEnergy;
		cell->energy -= radiationEnergy;
	}

	SimulationData* _data;
};

__device__ void correctCellConnections(int cellIndex, BlockProcessorForCluster& clusterProcess)
{
	CellData* newCell = &clusterProcess.newCells[cellIndex];
	int numConnections = newCell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		newCell->connections[i] = newCell->connections[i]->nextTimestep;
	}
}

__device__ void clusterMovement(SimulationData &data, int clusterIndex)
{
	__shared__ BlockProcessorForCluster clusterProcess;
	clusterProcess.init(data, clusterIndex);

	int startCellIndex;
	int endCellIndex;
	calcPartition(clusterProcess.getNumOrigCells(), threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	clusterProcess.processingCollision(startCellIndex, endCellIndex);
	clusterProcess.processingMovement(startCellIndex, endCellIndex);
	clusterProcess.processingRadiation(startCellIndex, endCellIndex);
/*
	__syncthreads();

	if (clusterProcess.atLeastOneCellDestroyed) {

	}
*/

	__syncthreads();

	if (threadIdx.x == 0) {
		*clusterProcess.newCluster = clusterProcess.clusterCopy;
	}

	int numOrigCells = clusterProcess.origCluster->numCells;
	if (threadIdx.x < numOrigCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			correctCellConnections(cellIndex, clusterProcess);
		}
	}

	__syncthreads();
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
	auto cell = getFromMap(toInt2(oldParticle->pos), data.cellMap1, data.size);
	if (cell) {
		auto nextCell = cell->nextTimestep;
		atomicAdd(&nextCell->energy, oldParticle->energy);
		return;
	}
	ParticleData *newParticle = data.particlesAC2.getNewElement();
	*newParticle = *oldParticle;
	newParticle->pos = add(newParticle->pos, newParticle->vel);
	mapPosCorrection(newParticle->pos, data.size);
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

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster.cells[cellIndex].absPos;
		setToMap<CellData>({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, nullptr, data.cellMap1, size);
	}
}

__device__ void clearParticle(SimulationData const &data, int particleIndex)
{
	auto const &particle = data.particlesAC1.getEntireArray()[particleIndex];
	setToMap<ParticleData>(toInt2(particle.pos), nullptr, data.particleMap1, data.size);
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
