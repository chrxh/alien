#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "CudaPhysics.cuh"
#include "Map.cuh"

class ParticleBuilder
{
public:
	__inline__ __device__ void init(SimulationDataInternal& data);

	__inline__ __device__ void processingDataCopy(int startParticleIndex, int endParticleIndex);

private:

	SimulationDataInternal* _data;
	Map<CellData> _cellMap;
	Map<ParticleData> _origParticleMap;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleBuilder::init(SimulationDataInternal & data)
{
	_data = &data;
	_cellMap.init(data.size, data.cellMap);
	_origParticleMap.init(data.size, data.particleMap);
}

__inline__ __device__ void ParticleBuilder::processingDataCopy(int startParticleIndex, int endParticleIndex)
{
	for (int particleIndex = startParticleIndex; particleIndex <= endParticleIndex; ++particleIndex) {
		ParticleData *origParticle = &_data->particlesAC1.getEntireArray()[particleIndex];
		if (!origParticle->alive) {
			continue;
		}
		if (auto cell = _cellMap.get(origParticle->pos)) {
			if (auto nextCell = cell->nextTimestep) {
				atomicAdd(&nextCell->energy, origParticle->energy);
				continue;
			}
		}
		ParticleData* newParticle = _data->particlesAC2.getNewElement();
		*newParticle = *origParticle;
	}
}
