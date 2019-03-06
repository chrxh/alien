#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ParticleReorganizer
{
public:
	__inline__ __device__ void init(SimulationData& data);

	__inline__ __device__ void processingDataCopy(int startParticleIndex, int endParticleIndex);

private:

	SimulationData* _data;
	Map<Cell> _cellMap;
	Map<Particle> _origParticleMap;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleReorganizer::init(SimulationData & data)
{
	_data = &data;
	_cellMap.init(data.size, data.cellMap);
	_origParticleMap.init(data.size, data.particleMap);
}

__inline__ __device__ void ParticleReorganizer::processingDataCopy(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		Particle *origParticle = &_data->particlesAC1.getEntireArray()[particleIndex];
		if (!origParticle->alive) {
			continue;
		}
		if (auto cell = _cellMap.get(origParticle->pos)) {
			if (auto nextCell = cell->nextTimestep) {
				if (nextCell->alive) {
					atomicAdd(&nextCell->energy, origParticle->energy);
					continue;
				}
			}
		}
		Particle* newParticle = _data->particlesAC2.getNewElement();
		*newParticle = *origParticle;
	}
}
