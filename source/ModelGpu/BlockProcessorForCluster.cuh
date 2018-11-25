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
	__inline__ __device__ void init(SimulationDataInternal& data, int clusterIndex);

	__inline__ __device__ void processingCollision(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDecomposition(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingMovement(int startCellIndex, int endCellIndex);

	__inline__ __device__ int getNumOrigCells() const;

private:
	__inline__ __device__ void copyAndMoveCell(int cellIndex);
	__inline__ __device__ void correctCellConnections(int cellIndex);
	__inline__ __device__ void cellRadiation(CellData *cell);
	__inline__ __device__ void createNewParticle(CellData *cell);

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;

	ClusterData _clusterCopy;
	ClusterData *_origCluster;
	ClusterData *_newCluster;
	CellData *_newCells;
	float _rotMatrix[2][2];
	CollisionData _collisionData;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void BlockProcessorForCluster::init(SimulationDataInternal& data, int clusterIndex)
{
	if (0 == threadIdx.x) {
		_data = &data;
		_origCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
		_clusterCopy = *_origCluster;
		_newCells = _data->cellsAC2.getNewSubarray(_clusterCopy.numCells);
		_newCluster = _data->clustersAC2.getNewElement();
		Physics::calcRotationMatrix(_clusterCopy.angle, _rotMatrix);
		_collisionData.init();
		_cellMap.init(data.size, data.cellMap1, data.cellMap2);
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
		Physics::getCollisionDataForCell(_cellMap, origCell, _collisionData);
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
			Physics::calcCollision(&_clusterCopy, entry, _cellMap);
		}

		_clusterCopy.angle += _clusterCopy.angularVel;
		Physics::angleCorrection(_clusterCopy.angle);
		_clusterCopy.pos = add(_clusterCopy.pos, _clusterCopy.vel);
		_cellMap.mapPosCorrection(_clusterCopy.pos);
		_clusterCopy.numCells = 0;
		_clusterCopy.cells = _newCells;
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingMovement(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		copyAndMoveCell(cellIndex);
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		correctCellConnections(cellIndex);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		*_newCluster = _clusterCopy;
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingDecomposition(int startCellIndex, int endCellIndex)
{
	if (_origCluster->decompositionRequired) {
		__shared__ bool changes;
		changes = false;

		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			_origCluster->cells[cellIndex].tag = cellIndex;
		}
		__syncthreads();
		while (changes) {
			for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
				CellData& cell = _origCluster->cells[cellIndex];
				if (cell.alive) {
					for (int i = 0; i < cell.numConnections; ++i) {
						CellData& otherCell = *cell.connections[i];
						if (otherCell.tag < cell.tag) {
							cell.tag = otherCell.tag;
							changes = true;
						}
					}
				}
			}
			__syncthreads();
			changes = false;
		}
	}
}

__inline__ __device__ void BlockProcessorForCluster::processingRadiation(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];
		cellRadiation(origCell);
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
	int newCellIndex = atomicAdd(&_clusterCopy.numCells, 1);
	CellData *newCell = &_newCells[newCellIndex];
	*newCell = cellCopy;
	_cellMap.setToNewMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell);

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
		cell->cluster->decompositionRequired = true;
	}
}

__inline__ __device__ void BlockProcessorForCluster::createNewParticle(CellData *cell)
{
	auto particle = _data->particlesAC2.getNewElement();
	auto &pos = cell->absPos;
	particle->id = _data->numberGen.newId_Kernel();
	particle->pos = { pos.x + _data->numberGen.random(2.0f) - 1.0f, pos.y + _data->numberGen.random(2.0f) - 1.0f };
	_cellMap.mapPosCorrection(particle->pos);
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
