#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ClusterDynamics
{
public:
	__inline__ __device__ void init(SimulationData& data, int clusterIndex);
	__inline__ __device__ int getNumCells() const;

	__inline__ __device__ void processingMovement(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingCollision(int startCellIndex, int endCellIndex);
	__inline__ __device__ void destroyCloseCell(int startCellIndex, int endCellIndex);
	__inline__ __device__ void processingRadiation(int startCellIndex, int endCellIndex);

private:
	__inline__ __device__ void destroyCloseCell(Cell *cell);
	__inline__ __device__ void destroyCloseCell(float2 const& pos, Cell *cell);
	__inline__ __device__ void cellRadiation(Cell *cell);
	__inline__ __device__ Particle* createNewParticle(float energy, float2 const& pos, float2 const& vel);
	__inline__ __device__ bool areConnectable(Cell *cell1, Cell *cell2);

	SimulationData* _data;
	Map<Cell> _cellMap;

	Cluster *_cluster;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ClusterDynamics::init(SimulationData & data, int clusterIndex)
{
	_data = &data;
	_cluster = &data.clustersAC1.getEntireArray()[clusterIndex];
	_cellMap.init(data.size, data.cellMap);
}

__inline__ __device__ int ClusterDynamics::getNumCells() const
{
	return _cluster->numCells;
}

__inline__ __device__ void ClusterDynamics::processingCollision(int startCellIndex, int endCellIndex)
{
	__shared__ Cluster* cluster;
	__shared__ Cluster* firstOtherCluster;
	__shared__ int firstOtherClusterId;
	__shared__ int numberOfCollidingCells;
	__shared__ float2 collisionCenterPos;
	__shared__ bool avoidCollision;
	enum CollisionState { ElasticCollision, Fusion };
	__shared__ CollisionState state;
	if (0 == threadIdx.x) {
		cluster = _cluster;
		firstOtherCluster = nullptr;
		firstOtherClusterId = 0;
		collisionCenterPos.x = 0;
		collisionCenterPos.y = 0;
		numberOfCollidingCells = 0;
		avoidCollision = false;
		state = CollisionState::ElasticCollision;
	}
	__syncthreads();

	//find colliding cluster
	for (auto index = startCellIndex; index <= endCellIndex; ++index) {
		Cell* cell = &cluster->cells[index];
		for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
			for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
				Cell* otherCell = _cellMap.get(add(cell->absPos, {dx, dy}));

				if (!otherCell || otherCell == cell) {
					continue;
				}
				if (cluster == otherCell->cluster) {
					continue;
				}
				if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
					continue;
				}
				if (!cell->alive || !otherCell->alive) {
					continue;
				}
				if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
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

	if (!firstOtherCluster) {
		__syncthreads();
		return;
	}

	__shared__ DoubleLock lock;
	if (0 == threadIdx.x) {
		lock.init(&cluster->locked, &firstOtherCluster->locked);
		lock.tryLock();
	}
	__syncthreads();
	if (!lock.isLocked()) {
		__syncthreads();
		return;
	}

	for (auto index = startCellIndex; index <= endCellIndex; ++index) {
		Cell* cell = &cluster->cells[index];
		for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
			for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
				Cell* otherCell = _cellMap.get(add(cell->absPos, { dx, dy }));
				if (!otherCell || otherCell == cell) {
					continue;
				}
				if (firstOtherCluster != otherCell->cluster) {
					continue;
				}
				if (!cell->alive || !otherCell->alive) {
					continue;
				}
				if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
					continue;
				}
				if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
					avoidCollision = true;
					break;
				}
				if (length(sub(cell->vel, otherCell->vel)) >= cudaSimulationParameters.cellFusionVelocity && areConnectable(cell, otherCell)) {
					state = CollisionState::Fusion;
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

	//also checking numberOfCollidingCells because it might happen that the colliding cells are not alive anymore... 
	if (avoidCollision || 0 == numberOfCollidingCells) {	
		if (0 == threadIdx.x) {
			lock.releaseLock();
		}
		__syncthreads();
		return;
	}

	if (CollisionState::Fusion == state) {
        if (nullptr == cluster->clusterToFuse && nullptr == firstOtherCluster->clusterToFuse
			/*&& !cluster->decompositionRequired && !firstOtherCluster->decompositionRequired*/) {
            for (auto index = startCellIndex; index <= endCellIndex; ++index) {
                Cell* cell = &cluster->cells[index];
				for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
					for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
						Cell* otherCell = _cellMap.get(add(cell->absPos, { dx, dy }));
						if (!otherCell || otherCell == cell) {
							continue;
						}
						if (firstOtherCluster != otherCell->cluster) {
							continue;
						}
						if (!cell->alive || !otherCell->alive) {
							continue;
						}
						if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
							continue;
						}
						if (length(sub(cell->vel, otherCell->vel)) < cudaSimulationParameters.cellFusionVelocity || !areConnectable(cell, otherCell)) {
							continue;
						}
						auto otherIndex = atomicAdd(&otherCell->numConnections, 1);
						if (otherIndex < otherCell->maxConnections) {
							auto index = cell->numConnections++;
							cell->connections[index] = otherCell;
							otherCell->connections[otherIndex] = cell;
						}
						else {
							atomicAdd(&otherCell->numConnections, -1);
						}
					}
				}
            }
			__syncthreads();
			if (0 == threadIdx.x) {
				cluster->clusterToFuse = firstOtherCluster;
				firstOtherCluster->clusterToFuse = cluster;
			}
        }
	}

	if (CollisionState::ElasticCollision == state) {
		__shared__ float2 rAPp;
		__shared__ float2 rBPp;
		__shared__ float2 outwardVector;
		__shared__ float2 n;
		if (0 == threadIdx.x) {
			collisionCenterPos = div(collisionCenterPos, numberOfCollidingCells);
			rAPp = { collisionCenterPos.x - cluster->pos.x, collisionCenterPos.y - cluster->pos.y };
			_cellMap.mapDisplacementCorrection(rAPp);
			rBPp = { collisionCenterPos.x - firstOtherCluster->pos.x, collisionCenterPos.y - firstOtherCluster->pos.y };
			_cellMap.mapDisplacementCorrection(rBPp);
			outwardVector = sub(
				Physics::tangentialVelocity(rBPp, firstOtherCluster->vel, firstOtherCluster->angularVel),
				Physics::tangentialVelocity(rAPp, cluster->vel, cluster->angularVel));
			Physics::rotateQuarterCounterClockwise(rAPp);
			Physics::rotateQuarterCounterClockwise(rBPp);
			n.x = 0.0f;
			n.y = 0.0f;
		}
		__syncthreads();

		for (auto index = startCellIndex; index <= endCellIndex; ++index) {
			Cell* cell = &cluster->cells[index];
			for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
				for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
					Cell* otherCell = _cellMap.get(add(cell->absPos, { dx, dy }));
					if (!otherCell || otherCell == cell) {
						continue;
					}
					if (firstOtherCluster != otherCell->cluster) {
						continue;
					}
					if (!cell->alive || !otherCell->alive) {
						continue;
					}
					if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
						continue;
					}
					float2 normal = Physics::calcNormalToCell(otherCell, outwardVector);
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
			Physics::calcCollision(cluster->vel, firstOtherCluster->vel, rAPp, rBPp, cluster->angularVel,
				firstOtherCluster->angularVel, n, cluster->angularMass, firstOtherCluster->angularMass,
				mA, mB, vA2, vB2, angularVelA2, angularVelB2);

			cluster->vel = vA2;
			cluster->angularVel = angularVelA2;
			firstOtherCluster->vel = vB2;
			firstOtherCluster->angularVel = angularVelB2;
		}
	}
	__syncthreads();

	if (0 == threadIdx.x) {
		lock.releaseLock();
	}
	__syncthreads();
}

__inline__ __device__ void ClusterDynamics::destroyCloseCell(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		Cell *origCell = &_cluster->cells[cellIndex];
		destroyCloseCell(origCell);
	}
	__syncthreads();
}

__inline__ __device__ void ClusterDynamics::processingMovement(int startCellIndex, int endCellIndex)
{
	__shared__ float rotMatrix[2][2];
	if (0 == threadIdx.x) {
		_cluster->angle += _cluster->angularVel;
		Physics::angleCorrection(_cluster->angle);
		_cluster->pos = add(_cluster->pos, _cluster->vel);
		_cellMap.mapPosCorrection(_cluster->pos);
		Physics::rotationMatrix(_cluster->angle, rotMatrix);
	}
	__syncthreads();


	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		Cell *cell = &_cluster->cells[cellIndex];

		float2 absPos;
		absPos.x = cell->relPos.x*rotMatrix[0][0] + cell->relPos.y*rotMatrix[0][1] + _cluster->pos.x;
		absPos.y = cell->relPos.x*rotMatrix[1][0] + cell->relPos.y*rotMatrix[1][1] + _cluster->pos.y;
		cell->absPos = absPos;

		auto r = sub(cell->absPos, _cluster->pos);
		_cellMap.mapDisplacementCorrection(r);
		auto newVel = Physics::tangentialVelocity(r, _cluster->vel, _cluster->angularVel);

		auto a = sub(newVel, cell->vel);
		if (length(a) > cudaSimulationParameters.cellMaxForce) {
			if (_data->numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
				cell->alive = false;
				_cluster->decompositionRequired = true;
				createNewParticle(cell->energy, cell->absPos, cell->vel);
			}
		}

		cell->vel = newVel;

		if (cell->protectionCounter > 0) {
			--cell->protectionCounter;
		}
		_cellMap.set(absPos, cell);
	}
	__syncthreads();
}

__inline__ __device__ void ClusterDynamics::processingRadiation(int startCellIndex, int endCellIndex)
{
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		Cell *cell = &_cluster->cells[cellIndex];
		cellRadiation(cell);
	}
	__syncthreads();
}

__inline__ __device__ void ClusterDynamics::destroyCloseCell(Cell * cell)
{
	if (cell->protectionCounter > 0) {
		return;
	}
	destroyCloseCell(cell->absPos, cell);
}

__inline__ __device__ void ClusterDynamics::destroyCloseCell(float2 const & pos, Cell * cell)
{
	Cell* mapCell = _cellMap.get(pos);
	if (!mapCell || mapCell == cell) {
		return;
	}

	Cluster* mapCluster = mapCell->cluster;
	auto distance = _cellMap.mapDistance(cell->absPos, mapCell->absPos);
	if (distance < cudaSimulationParameters.cellMinDistance) {
		Cluster* cluster = cell->cluster;
		if (mapCluster->numCells >= cluster->numCells) {
			cell->alive = false;
			cluster->decompositionRequired = true;
			createNewParticle(cell->energy, cell->absPos, cell->vel);
		}
		else {
			auto lockState = atomicExch(&mapCluster->locked, 1);
			if (0 == lockState) {
				mapCell->alive = false;
				mapCluster->decompositionRequired = true;
				mapCluster->locked = 0;
				createNewParticle(mapCell->energy, mapCell->absPos, mapCell->vel);
			}
		}
	}
}

__inline__ __device__ void ClusterDynamics::cellRadiation(Cell *cell)
{
	if (_data->numberGen.random() < cudaSimulationParameters.radiationProbability) {
		auto &pos = cell->absPos;
        float2 particlePos = {static_cast<int>(pos.x) + _data->numberGen.random(3) - 1.5f,
                              static_cast<int>(pos.y) + _data->numberGen.random(3) - 1.5f};
        _cellMap.mapPosCorrection(particlePos);
        float2 particleVel =
            add(mul(cell->vel, cudaSimulationParameters.radiationVelocityMultiplier),
                {(_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                 (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation});
        float radiationEnergy = powf(cell->energy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
        radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProbability;
		radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
		if (radiationEnergy > cell->energy - 1) {
			radiationEnergy = cell->energy - 1;
		}
		cell->energy -= radiationEnergy;
		createNewParticle(radiationEnergy, particlePos, particleVel);
	}
	if (cell->energy < cudaSimulationParameters.cellMinEnergy) {
		cell->alive = false;
		cell->cluster->decompositionRequired = true;
	}
}

__inline__ __device__ Particle* ClusterDynamics::createNewParticle(float energy, float2 const& pos, float2 const& vel)
{
	Particle* particle = _data->particlesAC1.getNewElement();
	particle->id = _data->numberGen.createNewId_kernel();
	particle->locked = 0;
	particle->alive = true;
	particle->energy = energy;
	particle->pos = pos;
	particle->vel = vel;
	return particle;
}

__inline__ __device__ bool ClusterDynamics::areConnectable(Cell * cell1, Cell * cell2)
{
	return cell1->numConnections < cell1->maxConnections && cell2->numConnections < cell2->maxConnections;
}
