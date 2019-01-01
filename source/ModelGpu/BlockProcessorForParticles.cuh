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
		ParticleData *oldParticle = &_data->particlesAC1.getEntireArray()[particleIndex];
		if (auto cell = _cellMap.getFromNewMap(oldParticle->pos)) {
			atomicAdd(&cell->energy, oldParticle->energy);
			continue;
		}
		ParticleData *newParticle = _data->particlesAC2.getNewElement();
		*newParticle = *oldParticle;
		newParticle->pos = add(newParticle->pos, newParticle->vel);
		_particleMap.mapPosCorrection(newParticle->pos);
		_particleMap.setToNewMap(newParticle->pos, newParticle);
	}
}
