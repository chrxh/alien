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
	__inline__ __device__ void processingCollision(int startParticleIndex, int endParticleIndex);	//uses orig particle map

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
		if (!origParticle->alive) {
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
		ParticleData* particle = &_data->particlesAC1.getEntireArray()[particleIndex];
		ParticleData* otherParticle = _particleMap.getFromOrigMap(particle->pos);
		if (otherParticle && otherParticle != particle) {
			int* particleLockAddr1 = &particle->locked;
			int* particleLockAddr2 = &otherParticle->locked;
			if (otherParticle->id < particle->id) {
				particleLockAddr1 = &otherParticle->locked;
				particleLockAddr2 = &particle->locked;
			}
			int particleLocked1 = static_cast<bool>(atomicExch(particleLockAddr1, 1));
			int particleLocked2 = static_cast<bool>(atomicExch(particleLockAddr2, 1));
			if (0 == particleLocked1 && 0 == particleLocked2) {
				if (particle->alive && otherParticle->alive) {
					float factor1 = particle->energy / (particle->energy + otherParticle->energy);
					float factor2 = 1.0f - factor1;
					particle->vel = add(mul(particle->vel, factor1), mul(otherParticle->vel, factor2));
					atomicAdd(&particle->energy, otherParticle->energy);
					atomicAdd(&otherParticle->energy, -otherParticle->energy);
					otherParticle->alive = false;
				}
			}
			if (0 == particleLocked1) {
				*particleLockAddr1 = 0;
			}
			if (0 == particleLocked2) {
				*particleLockAddr2 = 0;
			}
		}
	}
}
