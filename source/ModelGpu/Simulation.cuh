#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "Constants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

__shared__ ClusterData clusterCopy;
__shared__ ClusterData *oldCluster;
__shared__ ClusterData *newCluster;
__shared__ CellData *newCells;
__shared__ float rotMatrix[2][2];
__shared__ CollisionData collisionData;
__shared__ bool atLeastOneCellDestroyed;

__device__ void createNewParticle(SimulationData &data, CellData *cell)
{
	auto particle = data.particlesAC2.getElement();
	auto &pos = cell->absPos;
	particle->id = data.numberGen.newId_Kernel();
	particle->pos = { pos.x + data.numberGen.random(2.0f) - 1.0f, pos.y + data.numberGen.random(2.0f) - 1.0f };
	mapPosCorrection(particle->pos, data.size);
	particle->vel = { (data.numberGen.random()-0.5f) * RADIATION_VELOCITY_PERTURBATION, (data.numberGen.random() - 0.5f) * RADIATION_VELOCITY_PERTURBATION };
	float radiationEnergy = powf(cell->energy, RADIATION_EXPONENT) * RADIATION_FACTOR;
	radiationEnergy = radiationEnergy / RADIATION_PROB;
	radiationEnergy = 2 * radiationEnergy * data.numberGen.random();
	if (radiationEnergy > cell->energy - 1) {
		radiationEnergy = cell->energy - 1;
	}
	particle->energy = radiationEnergy;
	cell->energy -= radiationEnergy;
}

__device__ void cellRadiation(SimulationData &data, CellData *cell)
{
	if (data.numberGen.random() < RADIATION_PROB) {
		createNewParticle(data, cell);
	}
	if (cell->energy < CELL_MIN_ENERGY) {
		cell->alive = false;
	}
}

__device__ void performCollision(SimulationData &data)
{
	for (int i = 0; i < collisionData.numEntries; ++i) {
		CollisionEntry* entry = &collisionData.entries[i];
		float numCollisions = static_cast<float>(entry->numCollisions);
		entry->collisionPos.x /= numCollisions;
		entry->collisionPos.y /= numCollisions;
		entry->normalVec.x /= numCollisions;
		entry->normalVec.y /= numCollisions;
		calcCollision(&clusterCopy, entry, data.size);
	}

	clusterCopy.angle += clusterCopy.angularVel;
	angleCorrection(clusterCopy.angle);
	clusterCopy.pos = add(clusterCopy.pos, clusterCopy.vel);
	mapPosCorrection(clusterCopy.pos, data.size);
	clusterCopy.numCells = 0;
	clusterCopy.cells = newCells;
}

__device__ void calcRotationMatrix()
{
	float sinAngle = __sinf(clusterCopy.angle*DEG_TO_RAD);
	float cosAngle = __cosf(clusterCopy.angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = -sinAngle;
	rotMatrix[1][0] = sinAngle;
	rotMatrix[1][1] = cosAngle;
}

__device__ void copyCluster(SimulationData &data)
{
	clusterCopy = *oldCluster;
	newCells = data.cellsAC2.getArray_Kernel(clusterCopy.numCells);
	newCluster = data.clustersAC2.getElement();
}

__device__  CellData* copyMoveAndReturnCell(SimulationData &data, int &cellIndex)
{
	CellData *oldCell = &oldCluster->cells[cellIndex];
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
	setToMap<CellData>({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell, data.cellMap2, data.size);

	oldCell->nextTimestep = newCell;
	return newCell;
}

__device__ void correctCellConnections(int cellIndex)
{
	CellData* newCell = &newCells[cellIndex];
	int numConnections = newCell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		newCell->connections[i] = newCell->connections[i]->nextTimestep;
	}
}

__device__ void clusterMovement(SimulationData &data, int clusterIndex)
{

	oldCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
	int startCellIndex;
	int endCellIndex;
	int oldNumCells = oldCluster->numCells;
	if (threadIdx.x < oldNumCells) {

		calcPartition(oldCluster->numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

		if (threadIdx.x == 0) {
			copyCluster(data);
			calcRotationMatrix();
			atLeastOneCellDestroyed = false;
			collisionData.init();
		}
	}
	
	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData *oldCell = &oldCluster->cells[cellIndex];
			collectCollisionData(data, oldCell, collisionData);
			if (!oldCell->alive) {
				atLeastOneCellDestroyed = true;
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		performCollision(data);
	}

	__syncthreads();

	if (atLeastOneCellDestroyed) {

	}

	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData *newCell = copyMoveAndReturnCell(data, cellIndex);
			cellRadiation(data, newCell);
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		*newCluster = clusterCopy;
	}

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			correctCellConnections(cellIndex);
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
	ParticleData *newParticle = data.particlesAC2.getElement();
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
