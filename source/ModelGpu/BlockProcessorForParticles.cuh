#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "CudaPhysics.cuh"
#include "Map.cuh"

class BlockProcessorForParticles
{
public:
	__inline__ __device__ void init(SimulationDataInternal& data);
	
	__inline__ __device__ void processingMovement(int startParticleIndex, int endParticleIndex);
	__inline__ __device__ void processingCollision(int startParticleIndex, int endParticleIndex);

private:

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;
	Map<ParticleData> _particleMap;
};



/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void BlockProcessorForParticles::init(SimulationDataInternal & data)
{
	_data = &data;
	_cellMap.init(data.size, data.cellMap1, data.cellMap2);
	_particleMap.init(data.size, data.particleMap1, data.particleMap2);
}

__inline__ __device__ void BlockProcessorForParticles::processingMovement(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		ParticleData *origParticle = &_data->particlesAC1.getEntireArray()[particleIndex];
		if (0.0f == origParticle->energy) {
			continue;
		}
		if (auto cell = _cellMap.getFromNewMap(origParticle->pos)) {
			atomicAdd(&cell->energy, origParticle->energy);
			continue;
		}
		ParticleData* newParticle = _data->particlesAC2.getNewElement();
		*newParticle = *origParticle;
		newParticle->pos = add(origParticle->pos, origParticle->vel);
		_particleMap.mapPosCorrection(newParticle->pos);
		_particleMap.setToNewMap(newParticle->pos, newParticle);
	}
}

__inline__ __device__ void BlockProcessorForParticles::processingCollision(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		ParticleData* particle = &_data->particlesAC2.getEntireArray()[particleIndex];
		ParticleData* otherParticle = _particleMap.getFromNewMap(particle->pos);
		if (otherParticle && otherParticle != particle) {
			int particleLocked = static_cast<bool>(atomicExch((int*)(&particle->locked), 1));
			int otherParticleLocked = static_cast<bool>(atomicExch((int*)(&otherParticle->locked), 1));
			if (0 == particleLocked && 0 == otherParticleLocked) {
				float factor1 = particle->energy / (particle->energy + otherParticle->energy);
				float factor2 = 1.0f - factor1;
				particle->vel = add(mul(particle->vel, factor1), mul(otherParticle->vel, factor2));
				particle->energy += otherParticle->energy;
				otherParticle->energy = 0.0f;
				particle->locked = 0;
				otherParticle->locked = 0;
			}
		}
	}
}
