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
	__inline__ __device__ int getNumOrigCells() const;

	//synchronizing threads
	__inline__ __device__ void processingMovement(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDecomposition(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDataCopy(int startCellIndex, int endCellIndex);

private:
	//synchronizing threads
	__inline__ __device__ void processingDataCopyWithDecomposition(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDataCopyWithoutDecomposition(int startCellIndex, int endCellIndex);

	//not synchronizing threads
	__inline__ __device__ void copyAndUpdateCell(CellData *origCell, CellData* newCell, ClusterData* origCluster
		, ClusterData* newCluster, float (&rotMatrix)[2][2]);
	__inline__ __device__ void correctCellConnections(CellData *origCell);
	__inline__ __device__ void cellRadiation(CellData *cell);
	__inline__ __device__ void createNewParticle(CellData *cell);

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;

	ClusterData _newCluster;
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
		_newCluster = *_origCluster;
		_collisionData.init();
		_cellMap.init(data.size, data.cellMap1, data.cellMap2);
	}
	__syncthreads();
}

__inline__ __device__ int BlockProcessorForCluster::getNumOrigCells() const
{
	return _origCluster->numCells;
}

__inline__ __device__ void BlockProcessorForCluster::processingDataCopyWithDecomposition(int startCellIndex, int endCellIndex)
{
	__shared__ int numDecompositions;
	struct Entry {
		int tag;
		float rotMatrix[2][2];
		ClusterData cluster;
	};
	__shared__ Entry entries[MAX_DECOMPOSITIONS];

	numDecompositions = 0;
	for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
		entries[i].tag = -1;
		entries[i].cluster.pos = { 0.0f, 0.0f };
		entries[i].cluster.vel = _newCluster.vel;		//TODO: calculate
		entries[i].cluster.angle = _newCluster.angle;
		entries[i].cluster.angularVel = _newCluster.angularVel;		//TODO: calculate
		entries[i].cluster.angularMass = 0.0f;
		entries[i].cluster.numCells = 0;
		entries[i].cluster.decompositionRequired = false;
	}
	__syncthreads();
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData& cell = _origCluster->cells[cellIndex];
		bool foundMatch = false;
		for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
			int origTag = atomicCAS(&entries[index].tag, cell.tag, cell.tag);
			if (-1 == origTag) {	//use free 
				atomicAdd(&numDecompositions, 1);
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);
				entries[index].cluster.id = _data->numberGen.createNewId_kernel();
				Physics::calcRotationMatrix(entries[index].cluster.angle, entries[index].rotMatrix);
				foundMatch = true;
				break;
			}
			if (cell.tag == origTag) {	//matching entry
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);
				foundMatch = true;
				break;
			}
		}
		if (!foundMatch) {	//no match? use last entry
			cell.tag = entries[MAX_DECOMPOSITIONS - 1].tag;
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.numCells, 1);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.x, cell.absPos.x);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.y, cell.absPos.y);
			entries[MAX_DECOMPOSITIONS - 1].cluster.decompositionRequired = true;
		}
	}
	__syncthreads();

	int startDecompositionIndex;
	int endDecompositionIndex;
	__shared__ ClusterData* newClusters[MAX_DECOMPOSITIONS];
	calcPartition(numDecompositions, threadIdx.x, blockDim.x, startDecompositionIndex, endDecompositionIndex);
	for (int index = startDecompositionIndex; index <= endDecompositionIndex; ++index) {
		entries[index].cluster.pos.x /= entries[index].cluster.numCells;
		entries[index].cluster.pos.y /= entries[index].cluster.numCells;
		entries[index].cluster.cells = _data->cellsAC2.getNewSubarray(entries[index].cluster.numCells);
		entries[index].cluster.numCells = 0;

		newClusters[index] = _data->clustersAC2.getNewElement();
		*newClusters[index] = entries[index].cluster;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData& cell = _origCluster->cells[cellIndex];
		for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
			if (cell.tag == entries[index].tag) {
				ClusterData* newCluster = newClusters[index];
				float2 deltaPos = { cell.absPos.x - newCluster->pos.x, cell.absPos.y - newCluster->pos.y };
				newCluster->angularMass += deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y;

				int newCellIndex = atomicAdd(&newCluster->numCells, 1);
				CellData& newCell = newCluster->cells[newCellIndex];

				copyAndUpdateCell(&cell, &newCell, _origCluster, newCluster, entries[index].rotMatrix);

				float(&invRotMatrix)[2][2] = entries[index].rotMatrix;
				swap(invRotMatrix[0][1], invRotMatrix[1][0]);
				cell.relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
				cell.relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];
			}
		}
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData& cell = _origCluster->cells[cellIndex];
		correctCellConnections(cell.nextTimestep);
	}
	__syncthreads();

}

__inline__ __device__ void BlockProcessorForCluster::processingDataCopyWithoutDecomposition(int startCellIndex, int endCellIndex)
{
	__shared__ ClusterData* newCluster;
	__shared__ CellData* newCells;
	__shared__ float rotMatrix[2][2];

	if (threadIdx.x == 0) {
		newCluster = _data->clustersAC2.getNewElement();
		newCells = _data->cellsAC2.getNewSubarray(_newCluster.numCells);
		Physics::calcRotationMatrix(_newCluster.angle, rotMatrix);
		_newCluster.numCells = 0;
		_newCluster.cells = newCells;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];

		int newCellIndex = atomicAdd(&_newCluster.numCells, 1);
		CellData *newCell = &newCells[newCellIndex];
		copyAndUpdateCell(origCell, newCell, _origCluster, newCluster, rotMatrix);
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		*newCluster = _newCluster;
	}
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData* newCell = &newCells[cellIndex];
		correctCellConnections(newCell);
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingMovement(int startCellIndex, int endCellIndex)
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
			Physics::calcCollision(&_newCluster, entry, _cellMap);
		}

		_newCluster.angle += _newCluster.angularVel;
		Physics::angleCorrection(_newCluster.angle);
		_newCluster.pos = add(_newCluster.pos, _newCluster.vel);
		_cellMap.mapPosCorrection(_newCluster.pos);
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster::processingDataCopy(int startCellIndex, int endCellIndex)
{
	if (_origCluster->decompositionRequired) {
		processingDataCopyWithDecomposition(startCellIndex, endCellIndex);
	}
	else {
		processingDataCopyWithoutDecomposition(startCellIndex, endCellIndex);
	}
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
						if (otherCell.alive) {
							if (otherCell.tag < cell.tag) {
								cell.tag = otherCell.tag;
								changes = true;
							}
						}
						else {
							for (int j = i + 1; j < cell.numConnections; ++j) {
								cell.connections[j - 1] = cell.connections[j];
							}
							--cell.numConnections;
							--i;
						}
					}
				}
				else {
					cell.numConnections = 0;
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

__inline__ __device__  void BlockProcessorForCluster::copyAndUpdateCell(CellData *origCell, CellData* newCell
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
	particle->id = _data->numberGen.createNewId_kernel();
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
