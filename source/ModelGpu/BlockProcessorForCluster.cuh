#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "CudaPhysics.cuh"
#include "Map.cuh"

class BlockProcessorForCluster1
{
public:
	__inline__ __device__ void init(SimulationDataInternal& data, int clusterIndex);
	__inline__ __device__ int getNumOrigCells() const;

	//synchronizing threads
	__inline__ __device__ void processingMovementAndCollision(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDecomposition(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDataCopy(int startCellIndex, int endCellIndex);

private:
	//synchronizing threads
	__inline__ __device__ void processingDataCopyWithDecomposition(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingDataCopyWithoutDecomposition(int startCellIndex, int endCellIndex);

	//not synchronizing threads
	__inline__ __device__ void copyAndUpdateCell(CellData *origCell, CellData* newCell
		, ClusterData* newCluster, float2 const& clusterPos, float (&rotMatrix)[2][2]);
	__inline__ __device__ void correctCellConnections(CellData *origCell);
	__inline__ __device__ void cellRadiation(CellData *cell);
	__inline__ __device__ ParticleData* createNewParticle();
	__inline__ __device__ void killCloseCells(Map<CellData> const &map, CellData *cell);
	__inline__ __device__ void killCloseCell(float2 const& pos, Map<CellData> const &map, CellData *cell);

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;

	ClusterData _modifiedCluster;
	ClusterData *_origCluster;
	CollisionData _collisionData;
};

//experimental
class BlockProcessorForCluster2
{
public:
	__inline__ __device__ void init(SimulationDataInternal& data, int clusterIndex);
	__inline__ __device__ int getNumCells() const;

	//synchronizing threads
	__inline__ __device__ void processingCollision(int startCellIndex, int endCellIndex);	//uses new maps

private:

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;

	ClusterData *_cluster;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void BlockProcessorForCluster1::init(SimulationDataInternal& data, int clusterIndex)
{
	_data = &data;
	_origCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
	_modifiedCluster = *_origCluster;
	_collisionData.init();
	_cellMap.init(data.size, data.cellMap1, data.cellMap2);
}

__inline__ __device__ int BlockProcessorForCluster1::getNumOrigCells() const
{
	return _origCluster->numCells;
}

__inline__ __device__ void BlockProcessorForCluster1::processingDataCopyWithDecomposition(int startCellIndex, int endCellIndex)
{
	__shared__ int numDecompositions;
	struct Entry {
		int tag;
		float rotMatrix[2][2];
		float angularMassOfOrigCenter;
		ClusterData cluster;
	};
	__shared__ Entry entries[MAX_DECOMPOSITIONS];
	if (0 == threadIdx.x) {
		numDecompositions = 0;
		for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
			entries[i].tag = -1;
			entries[i].angularMassOfOrigCenter = 0.0f;
			entries[i].cluster.pos = { 0.0f, 0.0f };
			entries[i].cluster.vel = { 0.0f, 0.0f };
			entries[i].cluster.angle = _modifiedCluster.angle;
			entries[i].cluster.angularVel = 0.0f;			
			entries[i].cluster.angularMass = 0.0f;
			entries[i].cluster.numCells = 0;
			entries[i].cluster.decompositionRequired = false;
		}
	}
	__syncthreads();
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData& cell = _origCluster->cells[cellIndex];
		bool foundMatch = false;
		for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
			int origTag = atomicCAS(&entries[index].tag, -1, cell.tag);
			if (-1 == origTag) {	//use free 
				atomicAdd(&numDecompositions, 1);
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);

				auto tangentialVel = CudaPhysics::tangentialVelocity(cell.relPos, _origCluster->vel, _origCluster->angularVel);
				atomicAdd(&entries[index].cluster.vel.x, tangentialVel.x);
				atomicAdd(&entries[index].cluster.vel.y, tangentialVel.y);
				auto angularMass = cell.relPos.x * cell.relPos.x + cell.relPos.y * cell.relPos.y;
				atomicAdd(&entries[index].angularMassOfOrigCenter, angularMass);

				entries[index].cluster.id = _data->numberGen.createNewId_kernel();
				CudaPhysics::rotationMatrix(entries[index].cluster.angle, entries[index].rotMatrix);
				foundMatch = true;
				break;
			}
			if (cell.tag == origTag) {	//matching entry
				atomicAdd(&entries[index].cluster.numCells, 1);
				atomicAdd(&entries[index].cluster.pos.x, cell.absPos.x);
				atomicAdd(&entries[index].cluster.pos.y, cell.absPos.y);

				float2 tangentialVel = CudaPhysics::tangentialVelocity(cell.relPos, _origCluster->vel, _origCluster->angularVel);
				atomicAdd(&entries[index].cluster.vel.x, tangentialVel.x);
				atomicAdd(&entries[index].cluster.vel.y, tangentialVel.y);
				auto angularMass = cell.relPos.x * cell.relPos.x + cell.relPos.y * cell.relPos.y;
				atomicAdd(&entries[index].angularMassOfOrigCenter, angularMass);

				foundMatch = true;
				break;
			}
		}
		if (!foundMatch) {	//no match? use last entry
			cell.tag = entries[MAX_DECOMPOSITIONS - 1].tag;
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.numCells, 1);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.x, cell.absPos.x);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.y, cell.absPos.y);

			float2 tangentialVel = CudaPhysics::tangentialVelocity(cell.relPos, _origCluster->vel, _origCluster->angularVel);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.x, tangentialVel.x);
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.y, tangentialVel.y);
			auto angularMass = cell.relPos.x * cell.relPos.x + cell.relPos.y * cell.relPos.y;
			atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].angularMassOfOrigCenter, angularMass);

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
		entries[index].cluster.vel.x /= entries[index].cluster.numCells;
		entries[index].cluster.vel.y /= entries[index].cluster.numCells;
		entries[index].cluster.cells = _data->cellsAC2.getNewSubarray(entries[index].cluster.numCells);
		entries[index].cluster.numCells = 0;

		newClusters[index] = _data->clustersAC2.getNewElement();
		*newClusters[index] = entries[index].cluster;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData& cell = _origCluster->cells[cellIndex];
		for (int index = 0; index < numDecompositions; ++index) {
			if (cell.tag == entries[index].tag) {
				ClusterData* newCluster = newClusters[index];
				float2 deltaPos = { cell.absPos.x - newCluster->pos.x, cell.absPos.y - newCluster->pos.y };
				auto angularMass = deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y;
				atomicAdd(&newCluster->angularMass, angularMass);

				float2 tangentialVel = CudaPhysics::tangentialVelocity(cell.relPos, _origCluster->vel, _origCluster->angularVel);
				float2 relVel = { tangentialVel.x - newCluster->vel.x, tangentialVel.y - newCluster->vel.y };
				float angularMomentum = CudaPhysics::angularMomentum(cell.relPos, relVel);
				float angularVel = CudaPhysics::angularVelocity(angularMomentum, entries[index].angularMassOfOrigCenter);
				atomicAdd(&newCluster->angularVel, angularVel);

				float(&invRotMatrix)[2][2] = entries[index].rotMatrix;
				swap(invRotMatrix[0][1], invRotMatrix[1][0]);
				cell.relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
				cell.relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];
				swap(invRotMatrix[0][1], invRotMatrix[1][0]);

				int newCellIndex = atomicAdd(&newCluster->numCells, 1);
				CellData& newCell = newCluster->cells[newCellIndex];
				copyAndUpdateCell(&cell, &newCell, newCluster, newCluster->pos, entries[index].rotMatrix);
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

__inline__ __device__ void BlockProcessorForCluster1::processingDataCopyWithoutDecomposition(int startCellIndex, int endCellIndex)
{
	__shared__ ClusterData* newCluster;
	__shared__ CellData* newCells;
	__shared__ float rotMatrix[2][2];

	if (threadIdx.x == 0) {
		newCluster = _data->clustersAC2.getNewElement();
		newCells = _data->cellsAC2.getNewSubarray(_modifiedCluster.numCells);
		CudaPhysics::rotationMatrix(_modifiedCluster.angle, rotMatrix);
		_modifiedCluster.numCells = 0;
		_modifiedCluster.cells = newCells;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];

		int newCellIndex = atomicAdd(&_modifiedCluster.numCells, 1);
		CellData *newCell = &newCells[newCellIndex];
		copyAndUpdateCell(origCell, newCell, newCluster, _modifiedCluster.pos, rotMatrix);
	}

	__syncthreads();
	if (threadIdx.x == 0) {
		*newCluster = _modifiedCluster;
	}
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData* newCell = &newCells[cellIndex];
		correctCellConnections(newCell);
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster1::processingMovementAndCollision(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];
/*
		CudaPhysics::getCollisionDataForCell(_cellMap, origCell, _collisionData);
*/
		killCloseCells(_cellMap, origCell);
	}

	__syncthreads();

	if (0 == threadIdx.x) {
/*
		for (int i = 0; i < _collisionData.numEntries; ++i) {
			CollisionEntry* entry = &_collisionData.entries[i];
			float numCollisions = static_cast<float>(entry->numCollisions);
			entry->collisionPos.x /= numCollisions;
			entry->collisionPos.y /= numCollisions;
			entry->normalVec.x /= numCollisions;
			entry->normalVec.y /= numCollisions;
			CudaPhysics::calcCollision(&_modifiedCluster, entry, _cellMap);
		}
*/

		_modifiedCluster.angle += _modifiedCluster.angularVel;
		CudaPhysics::angleCorrection(_modifiedCluster.angle);
		_modifiedCluster.pos = add(_modifiedCluster.pos, _modifiedCluster.vel);
		_cellMap.mapPosCorrection(_modifiedCluster.pos);
	}
	__syncthreads();
}

__inline__ __device__ void BlockProcessorForCluster1::processingDataCopy(int startCellIndex, int endCellIndex)
{
	if (_origCluster->numCells == 1 && !_origCluster->cells[0].alive) {
		__syncthreads();
		return;
	}
	if (_origCluster->decompositionRequired) {
		processingDataCopyWithDecomposition(startCellIndex, endCellIndex);
	}
	else {
		processingDataCopyWithoutDecomposition(startCellIndex, endCellIndex);
	}
}

__inline__ __device__ void BlockProcessorForCluster1::processingDecomposition(int startCellIndex, int endCellIndex)
{
	if (_origCluster->decompositionRequired) {
		__shared__ bool changes;

		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			_origCluster->cells[cellIndex].tag = cellIndex;
		}
		do {
			changes = false;
			__syncthreads();
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
		} while (changes);
		__syncthreads();
	}
}

__inline__ __device__ void BlockProcessorForCluster1::processingRadiation(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellData *origCell = &_origCluster->cells[cellIndex];
		cellRadiation(origCell);
	}
	__syncthreads();
}

__inline__ __device__  void BlockProcessorForCluster1::copyAndUpdateCell(CellData *origCell, CellData* newCell
	, ClusterData* newCluster, float2 const& clusterPos, float (&rotMatrix)[2][2])
{
	CellData cellCopy = *origCell;

	float2 absPos;
	absPos.x = cellCopy.relPos.x*rotMatrix[0][0] + cellCopy.relPos.y*rotMatrix[0][1] + clusterPos.x;
	absPos.y = cellCopy.relPos.x*rotMatrix[1][0] + cellCopy.relPos.y*rotMatrix[1][1] + clusterPos.y;
	cellCopy.absPos = absPos;
	cellCopy.cluster = newCluster;
	if (cellCopy.protectionCounter > 0) {
		--cellCopy.protectionCounter;
	}
	*newCell = cellCopy;
	_cellMap.setToNewMap(absPos, newCell);

	origCell->nextTimestep = newCell;
}

__inline__ __device__ void BlockProcessorForCluster1::correctCellConnections(CellData* cell)
{
	int numConnections = cell->numConnections;
	for (int i = 0; i < numConnections; ++i) {
		cell->connections[i] = cell->connections[i]->nextTimestep;
	}
}

__inline__ __device__ void BlockProcessorForCluster1::cellRadiation(CellData *cell)
{
	if (_data->numberGen.random() < cudaSimulationParameters.radiationProbability) {
		auto particle = createNewParticle();
		auto &pos = cell->absPos;
		particle->pos = { static_cast<int>(pos.x) + 0.5f + _data->numberGen.random(2.0f) - 1.0f, static_cast<int>(pos.y) + 0.5f + _data->numberGen.random(2.0f) - 1.0f };
		_cellMap.mapPosCorrection(particle->pos);
		particle->vel = { (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation
			, (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation };
		float radiationEnergy = powf(cell->energy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
		radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProbability;
		radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
		if (radiationEnergy > cell->energy - 1) {
			radiationEnergy = cell->energy - 1;
		}
		particle->energy = radiationEnergy;
		cell->energy -= radiationEnergy;
	}
	if (cell->energy < cudaSimulationParameters.cellMinEnergy) {
		cell->alive = false;
		cell->cluster->decompositionRequired = true;
	}
	if (1 == cell->cluster->numCells && !cell->alive) {
		auto particle = createNewParticle();
		particle->pos = cell->absPos;
		particle->vel = cell->cluster->vel;
		particle->energy = cell->energy;
	}
}

__inline__ __device__ ParticleData* BlockProcessorForCluster1::createNewParticle()
{
	ParticleData* particle = _data->particlesAC2.getNewElement();
	particle->id = _data->numberGen.createNewId_kernel();
	particle->locked = 0;
	particle->alive = true;
	return particle;
}

__inline__ __device__ void BlockProcessorForCluster1::killCloseCells(Map<CellData> const & map, CellData * cell)
{
	if (cell->protectionCounter > 0) {
		return;
	}

	float2 absPos = cell->absPos;
/*
	killCloseCell(absPos, map, cell);
*/

	--absPos.x;
	--absPos.y;
	killCloseCell(absPos, map, cell);
	++absPos.x;
	killCloseCell(absPos, map, cell);
	++absPos.x;
	killCloseCell(absPos, map, cell);

	++absPos.y;
	absPos.x -= 2;
	killCloseCell(absPos, map, cell);
	absPos.x += 2;
	killCloseCell(absPos, map, cell);

	++absPos.y;
	absPos.x -= 2;
	killCloseCell(absPos, map, cell);
	++absPos.x;
	killCloseCell(absPos, map, cell);
	++absPos.x;
	killCloseCell(absPos, map, cell);
}

__inline__ __device__ void BlockProcessorForCluster1::killCloseCell(float2 const & pos, Map<CellData> const & map, CellData * cell)
{
	CellData* mapCell = map.getFromOrigMap(pos);
	if (!mapCell || mapCell == cell) {
		return;
	}

	ClusterData* mapCluster = mapCell->cluster;
	auto distanceSquared = map.mapDistanceSquared(cell->absPos, mapCell->absPos);
	if (distanceSquared < cudaSimulationParameters.cellMinDistance * cudaSimulationParameters.cellMinDistance) {
		ClusterData* cluster = cell->cluster;
		if (mapCluster->numCells >= cluster->numCells) {
			cell->alive = false;
			_modifiedCluster.decompositionRequired = true;
		}
	}
}

__inline__ __device__ void BlockProcessorForCluster2::init(SimulationDataInternal & data, int clusterIndex)
{
	_data = &data;
	_cluster = &data.clustersAC2.getEntireArray()[clusterIndex];
	_cellMap.init(data.size, data.cellMap1, data.cellMap2);
}

__inline__ __device__ int BlockProcessorForCluster2::getNumCells() const
{
	return _cluster->numCells;
}

__inline__ __device__ void BlockProcessorForCluster2::processingCollision(int startCellIndex, int endCellIndex)
{
	__shared__ ClusterData* cluster;
	__shared__ ClusterData* firstOtherCluster;
	__shared__ int firstOtherClusterId;
	__shared__ int clusterLocked1;
	__shared__ int clusterLocked2;
	__shared__ int* clusterLockAddr1;
	__shared__ int* clusterLockAddr2;
	__shared__ int numberOfCollidingCells;
	__shared__ float2 collisionCenterPos;
	__shared__ bool avoidCollision;
	if (0 == threadIdx.x) {
		cluster = _cluster;
		firstOtherCluster = nullptr;
		firstOtherClusterId = 0;
		clusterLocked1 = 1;
		clusterLocked2 = 1;
		collisionCenterPos.x = 0;
		collisionCenterPos.y = 0;
		numberOfCollidingCells = 0;
		avoidCollision = false;
	}
	__syncthreads();

	//find colliding cluster
	for (int index = startCellIndex; index <= endCellIndex; ++index) {
		CellData* cell = &cluster->cells[index];
		for (float dx = -1.0f; dx < 2.0f; ++dx) {
			for (float dy = -1.0f; dy < 2.0f; ++dy) {
				CellData* otherCell = _cellMap.getFromNewMap(add(cell->absPos, {dx, dy}));

				if (!otherCell || otherCell == cell) {
					continue;
				}
				if (cluster == otherCell->cluster) {
					continue;
				}
				if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
					continue;
				}
				int origFirstOtherClusterId = atomicCAS(&firstOtherClusterId, 0, otherCell->cluster->id);
				if (0 != origFirstOtherClusterId) {
					continue;
				}

				firstOtherCluster = otherCell->cluster;
			}
		}
	}
	__syncthreads();

	if (firstOtherCluster) {

		if (0 == threadIdx.x) {
			clusterLockAddr1 = &cluster->locked;
			clusterLockAddr2 = &firstOtherCluster->locked;
			if (firstOtherCluster->id < cluster->id) {
				clusterLockAddr1 = &firstOtherCluster->locked;
				clusterLockAddr2 = &cluster->locked;
			}
			clusterLocked1 = atomicExch(clusterLockAddr1, 1);
			if (0 == clusterLocked1) {
				clusterLocked2 = atomicExch(clusterLockAddr2, 1);
			}
		}
		__syncthreads();

		if (0 == clusterLocked1 && 0 == clusterLocked2) {
			for (int index = startCellIndex; index <= endCellIndex; ++index) {
				CellData* cell = &cluster->cells[index];
				for (float dx = -1.0f; dx < 2.0f; ++dx) {
					for (float dy = -1.0f; dy < 2.0f; ++dy) {
						CellData* otherCell = _cellMap.getFromNewMap(add(cell->absPos, { dx, dy }));
						if (!otherCell || otherCell == cell) {
							continue;
						}
						if (firstOtherCluster != otherCell->cluster) {
							continue;
						}
						if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
							avoidCollision = true;
							break;
						}

						atomicAdd(&collisionCenterPos.x, otherCell->absPos.x);
						atomicAdd(&collisionCenterPos.y, otherCell->absPos.y);
						atomicAdd(&numberOfCollidingCells, 1);
					}
					if (avoidCollision) {
						break;
					}
				}
				if (avoidCollision) {
					break;
				}
			}
			__syncthreads();

			if (!avoidCollision) {
				__shared__ float2 rAPp;
				__shared__ float2 rBPp;
				__shared__ float2 outwardVector;
				__shared__ float2 n;
				if (0 == threadIdx.x) {
					collisionCenterPos = div(collisionCenterPos, numberOfCollidingCells);
					rAPp = { collisionCenterPos.x - cluster->pos.x, collisionCenterPos.y - cluster->pos.y };
					// 				metric->correctDisplacement(rAPp);
					rBPp = { collisionCenterPos.x - firstOtherCluster->pos.x, collisionCenterPos.y - firstOtherCluster->pos.y };
					// 				metric->correctDisplacement(rBPp);
					outwardVector = sub(
						CudaPhysics::tangentialVelocity(rBPp, firstOtherCluster->vel, firstOtherCluster->angularVel),
						CudaPhysics::tangentialVelocity(rAPp, cluster->vel, cluster->angularVel));
					CudaPhysics::rotateQuarterCounterClockwise(rAPp);
					CudaPhysics::rotateQuarterCounterClockwise(rBPp);
					n.x = 0.0f;
					n.y = 0.0f;
				}
				__syncthreads();

				for (int index = startCellIndex; index <= endCellIndex; ++index) {
					CellData* cell = &cluster->cells[index];
					for (float dx = -1.0f; dx < 2.0f; ++dx) {
						for (float dy = -1.0f; dy < 2.0f; ++dy) {
							CellData* otherCell = _cellMap.getFromNewMap(add(cell->absPos, { dx, dy }));
							if (!otherCell || otherCell == cell) {
								continue;
							}
							if (firstOtherCluster != otherCell->cluster) {
								continue;
							}
							float2 normal = CudaPhysics::calcNormalToCell(otherCell, outwardVector);
							atomicAdd(&n.x, normal.x);
							atomicAdd(&n.y, normal.y);
							cell->protectionCounter = PROTECTION_TIMESTEPS;
							otherCell->protectionCounter = PROTECTION_TIMESTEPS;
						}
					}
				}
				__syncthreads();

				if (0 == threadIdx.x) {
					float mA = cluster->numCells;
					float mB = firstOtherCluster->numCells;
					float2 vA2{ 0.0f, 0.0f };
					float2 vB2{ 0.0f, 0.0f };
					float angularVelA2{ 0.0f };
					float angularVelB2{ 0.0f };
					normalize(n);
					CudaPhysics::calcCollision(cluster->vel, firstOtherCluster->vel, rAPp, rBPp, cluster->angularVel,
						firstOtherCluster->angularVel, n, cluster->angularMass, firstOtherCluster->angularMass,
						mA, mB, vA2, vB2, angularVelA2, angularVelB2);

					cluster->vel = vA2;
					cluster->angularVel = angularVelA2;
					firstOtherCluster->vel = vB2;
					firstOtherCluster->angularVel = angularVelB2;
				}
			}
		}
		__syncthreads();

		if (0 == threadIdx.x) {
			if (0 == clusterLocked1) {
				*clusterLockAddr1 = 0;
			}
			if (0 == clusterLocked2) {
				*clusterLockAddr2 = 0;
			}
		}
	}
	__syncthreads();
}
