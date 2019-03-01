#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "CudaPhysics.cuh"
#include "Map.cuh"

class ParticleDynamics
{
public:
	__inline__ __device__ void init(SimulationDataInternal& data);

	__inline__ __device__ void processingMovement(int startParticleIndex, int endParticleIndex);
	__inline__ __device__ void processingCollision(int startParticleIndex, int endParticleIndex);

private:

	SimulationDataInternal* _data;
	Map<ParticleData> _origParticleMap;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleDynamics::init(SimulationDataInternal & data)
{
	_data = &data;
	_origParticleMap.init(data.size, data.particleMap);
}

__inline__ __device__ void ParticleDynamics::processingMovement(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		ParticleData* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		particle->pos = add(particle->pos, particle->vel);
		_origParticleMap.mapPosCorrection(particle->pos);
		_origParticleMap.set(particle->pos, particle);
	}
}

__inline__ __device__ void ParticleDynamics::processingCollision(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		ParticleData* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		ParticleData* otherParticle = _origParticleMap.get(particle->pos);
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
