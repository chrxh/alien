#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ParticleDynamics
{
public:
	__inline__ __device__ void init(SimulationData& data);

	__inline__ __device__ void processingMovement(int startParticleIndex, int endParticleIndex);
	__inline__ __device__ void processingCollision(int startParticleIndex, int endParticleIndex);
	__inline__ __device__ void processingTransformation(int startParticleIndex, int endParticleIndex);

private:
	__inline__ __device__ void createRandomCell(float energy, float2 const& pos, float2 const& vel);

	SimulationData* _data;
	Map<Particle> _origParticleMap;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleDynamics::init(SimulationData & data)
{
	_data = &data;
	_origParticleMap.init(data.size, data.particleMap);
}

__inline__ __device__ void ParticleDynamics::processingMovement(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		Particle* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		particle->pos = add(particle->pos, particle->vel);
		_origParticleMap.mapPosCorrection(particle->pos);
		_origParticleMap.set(particle->pos, particle);
	}
}

__inline__ __device__ void ParticleDynamics::processingCollision(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		Particle* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		Particle* otherParticle = _origParticleMap.get(particle->pos);
		if (otherParticle && otherParticle != particle) {
			if (particle->alive && otherParticle->alive) {

				DoubleLock lock;
				lock.init(&particle->locked, &otherParticle->locked);
				lock.tryLock();
				if (!lock.isLocked()) {
					continue;
				}

				float factor1 = particle->energy / (particle->energy + otherParticle->energy);
				float factor2 = 1.0f - factor1;
				particle->vel = add(mul(particle->vel, factor1), mul(otherParticle->vel, factor2));
				atomicAdd(&particle->energy, otherParticle->energy);
				atomicAdd(&otherParticle->energy, -otherParticle->energy);
				otherParticle->alive = false;

				lock.releaseLock();
			}
		}
	}
}

__inline__ __device__ void ParticleDynamics::processingTransformation(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		Particle* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		auto innerEnergy = particle->energy - Physics::linearKineticEnergy(cudaSimulationParameters.cellMass, particle->vel);
		if (innerEnergy >= cudaSimulationParameters.cellMinEnergy) {
			if (_data->numberGen.random() < cudaSimulationParameters.cellTransformationProb) {
				createRandomCell(innerEnergy, particle->pos, particle->vel);
				particle->alive = false;
			}
		}
	}
}

__inline__ __device__ void ParticleDynamics::createRandomCell(float energy, float2 const & pos, float2 const & vel)
{
	Cluster* cluster = _data->clustersAC2.getNewElement();
	Cell* cell = _data->cellsAC2.getNewElement();

	cluster->id = _data->numberGen.createNewId_kernel();
	cluster->pos = pos;
	cluster->vel = vel;
	cluster->angle = 0.0f;
	cluster->angularVel = 0.0f;
	cluster->angularMass = 0.0f;
	cluster->numCells = 1;
	cluster->cells = cell;

	cluster->clusterToFuse = nullptr;
	cluster->locked = 0;
	cluster->decompositionRequired = false;

	cell->id = _data->numberGen.createNewId_kernel();
	cell->absPos = pos;
	cell->relPos = { 0.0f, 0.0f };
	cell->vel = vel;
	cell->energy = energy;
	cell->maxConnections = _data->numberGen.random(MAX_CELL_BONDS);
	cell->cluster = cluster;
	cell->numConnections = 0;
	cell->nextTimestep = nullptr;
	cell->alive = true;
	cell->protectionCounter = 0;
}
