#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "Constants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class BlockProcessorForCluster
{
public:
	__inline__ __device__ void init(SimulationData& data, int clusterIndex);

	__inline__ __device__ void processingCollision(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingMovement(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex);

	__inline__ __device__ int getNumOrigCells() const;

private:
	__inline__ __device__ void copyAndMoveCell(int cellIndex);
	__inline__ __device__ void correctCellConnections(int cellIndex);
	__inline__ __device__ void cellRadiation(CellData *cell);
	__inline__ __device__ void createNewParticle(CellData *cell);

	SimulationData* _data;
	ClusterData _clusterCopy;
	ClusterData *_origCluster;
	ClusterData *_newCluster;
	CellData *_newCells;
	float _rotMatrix[2][2];
	CollisionData _collisionData;
	bool _atLeastOneCellDestroyed;
};


/************************************************************************/
/* Implementations                                                      */
/************************************************************************/
__inline__ __device__ void BlockProcessorForCluster::init(SimulationData& data, int clusterIndex)
{
	if (0 == threadIdx.x) {
		_data = &data;
		_origCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
		_clusterCopy = *_origCluster;
		_newCells = _data->cellsAC2.getNewSubarray(_clusterCopy.numCells);
		_newCluster = _data->clustersAC2.getNewElement();
		calcRotationMatrix(_clusterCopy.angle, _rotMatrix);
		_atLeastOneCellDestroyed = false;
		_collisionData.init();
	}
	__syncthreads();
}

__inline__ __device__ int BlockProcessorForCluster::getNumOrigCells() const
{
	return _origCluster->numCells;
}

__inline__ __device__ void BlockProcessorForCluster::processingCollision(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];
		collectCollisionDataForCell(*_data, origCell, _collisionData);
		if (!origCell->alive) {
			_atLeastOneCellDestroyed = true;
		}
	}

	__syncthreads();

	if (0 == threadIdx.x) {

		for (int i = 0; i < _collisionData.numEntries; ++i) {
			CollisionEntry* entry = &_collisionData.entries[i];
			float numCollisions = static_cast<float>(entry->numCollisions);
			entry->collisionPos.x /= numCollisions;
			entry->collisionPos.y /= numCollisions;
			entry->normalVec.x /= numCollisions;
			entry->normalVec.y /= numCollisions;
			calcCollision(&_clusterCopy, entry, _data->size);
		}

		_clusterCopy.angle += _clusterCopy.angularVel;
		angleCorrection(_clusterCopy.angle);
		_clusterCopy.pos = add(_clusterCopy.pos, _clusterCopy.vel);
		mapPosCorrection(_clusterCopy.pos, _data->size);
		_clusterCopy.numCells = 0;
		_clusterCopy.cells = _newCells;
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingMovement(int startCellIndex, int endCellIndex)
{
	int numOrigCells = _origCluster->numCells;
	if (threadIdx.x < numOrigCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			copyAndMoveCell(cellIndex);
		}
	}
	__syncthreads();

	if (threadIdx.x < numOrigCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			correctCellConnections(cellIndex);
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		*_newCluster = _clusterCopy;
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingRadiation(int startCellIndex, int endCellIndex)
{
	int numOrigCells = _origCluster->numCells;
	if (threadIdx.x < numOrigCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData *origCell = &_origCluster->cells[cellIndex];
			cellRadiation(origCell);
		}
	}
	__syncthreads();
}

__inline__ __device__  void BlockProcessorForCluster::copyAndMoveCell(int cellIndex)
{
	CellData *origCell = &_origCluster->cells[cellIndex];
	CellData cellCopy = *origCell;

	float2 absPos;
	absPos.x = cellCopy.relPos.x*_rotMatrix[0][0] + cellCopy.relPos.y*_rotMatrix[0][1] + _clusterCopy.pos.x;
	absPos.y = cellCopy.relPos.x*_rotMatrix[1][0] + cellCopy.relPos.y*_rotMatrix[1][1] + _clusterCopy.pos.y;
	cellCopy.absPos = absPos;
	cellCopy.cluster = _newCluster;
	if (cellCopy.protectionCounter > 0) {
		--cellCopy.protectionCounter;
	}
	if (cellCopy.setProtectionCounterForNextTimestep) {
		cellCopy.protectionCounter = PROTECTION_TIMESTEPS;
		cellCopy.setProtectionCounterForNextTimestep = false;
	}
	int newCellIndex = atomicAdd(&_clusterCopy.numCells, 1);
	CellData *newCell = &_newCells[newCellIndex];
	*newCell = cellCopy;
	setToMap<CellData>({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell, _data->cellMap2, _data->size);

	origCell->nextTimestep = newCell;
}

__inline__ __device__ void BlockProcessorForCluster::correctCellConnections(int cellIndex)
{
	CellData* newCell = &_newCells[cellIndex];
	int numConnections = newCell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		newCell->connections[i] = newCell->connections[i]->nextTimestep;
	}
}

__inline__ __device__ void BlockProcessorForCluster::cellRadiation(CellData *cell)
{
	if (_data->numberGen.random() < RADIATION_PROB) {
		createNewParticle(cell);
	}
	if (cell->energy < CELL_MIN_ENERGY) {
		cell->alive = false;
	}
}

__inline__ __device__ void BlockProcessorForCluster::createNewParticle(CellData *cell)
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
