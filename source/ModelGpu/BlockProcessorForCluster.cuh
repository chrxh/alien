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
	__inline__ __device__ void copyAndMoveCell(CellData *origCell, CellData* newCell, ClusterData* origCluster
		, ClusterData* newCluster, float (&rotMatrix)[2][2]);
	__inline__ __device__ void correctCellConnections(CellData *origCell);
	__inline__ __device__ void cellRadiation(CellData *cell);
	__inline__ __device__ void createNewParticle(CellData *cell);

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;

	ClusterData _origClusterCopy;
	ClusterData *_origCluster;
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
		_origClusterCopy = *_origCluster;
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
			Physics::calcCollision(&_origClusterCopy, entry, _cellMap);
		}

		_origClusterCopy.angle += _origClusterCopy.angularVel;
		Physics::angleCorrection(_origClusterCopy.angle);
		_origClusterCopy.pos = add(_origClusterCopy.pos, _origClusterCopy.vel);
		_cellMap.mapPosCorrection(_origClusterCopy.pos);
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingMovement(int startCellIndex, int endCellIndex)
{
/*
	if (_origCluster->decompositionRequired) {
		__shared__ int numDecompositions;
		struct Entry {
			int tag;
			int numCells;
		};
		__shared__ Entry entries[MAX_DECOMPOSITIONS];

		numDecompositions = 0;
		for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
			entries[i].tag = -1;
			entries[i].numCells = 0;
		}
		__syncthreads();
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData& cell = _origCluster->cells[cellIndex];
			bool foundMatch = false;
			for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
				int origTag = atomicCAS(&entries[index].tag, cell.tag, cell.tag);
				if (-1 == origTag || cell.tag == origTag) {	//use free or matching entry
					atomicAdd(&numDecompositions, 1);
					atomicAdd(&entries[index].numCells, 1);
					foundMatch = true;
					break;
				}
			}
			if (!foundMatch) {	//no match? use last entry
				cell.tag = entries[MAX_DECOMPOSITIONS - 1].tag;
				atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].numCells, 1);
			}
		}
	}
	else {
*/
		__shared__ ClusterData* newCluster;
		__shared__ CellData* newCells;
		__shared__ float rotMatrix[2][2];

		if (threadIdx.x == 0) {
			newCluster = _data->clustersAC2.getNewElement();
			newCells = _data->cellsAC2.getNewSubarray(_origClusterCopy.numCells);
			Physics::calcRotationMatrix(_origClusterCopy.angle, rotMatrix);
			_origClusterCopy.numCells = 0;
			_origClusterCopy.cells = newCells;
		}
		__syncthreads();

		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData *origCell = &_origCluster->cells[cellIndex];

			int newCellIndex = atomicAdd(&_origClusterCopy.numCells, 1);
			CellData *newCell = &newCells[newCellIndex];
			copyAndMoveCell(origCell, newCell, _origCluster, newCluster, rotMatrix);
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			*newCluster = _origClusterCopy;
		}
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellData* newCell = &newCells[cellIndex];
			correctCellConnections(newCell);
		}
		__syncthreads();
/*
	}
*/
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

__inline__ __device__  void BlockProcessorForCluster::copyAndMoveCell(CellData *origCell, CellData* newCell
	, ClusterData* origCluster, ClusterData* newCluster, float (&rotMatrix)[2][2])
{
	CellData cellCopy = *origCell;

	float2 absPos;
	absPos.x = cellCopy.relPos.x*rotMatrix[0][0] + cellCopy.relPos.y*rotMatrix[0][1] + origCluster->pos.x;
	absPos.y = cellCopy.relPos.x*rotMatrix[1][0] + cellCopy.relPos.y*rotMatrix[1][1] + origCluster->pos.y;
	cellCopy.absPos = absPos;
	cellCopy.cluster = newCluster;
	if (cellCopy.protectionCounter > 0) {
		--cellCopy.protectionCounter;
	}
	*newCell = cellCopy;
	_cellMap.setToNewMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell);

	origCell->nextTimestep = newCell;
}

__inline__ __device__ void BlockProcessorForCluster::correctCellConnections(CellData* cell)
{
	int numConnections = cell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		cell->connections[i] = cell->connections[i]->nextTimestep;
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
