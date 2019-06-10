#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ParticleProcessorOnCopyData
{
public:
	__inline__ __device__ void init(SimulationData& data);

	__inline__ __device__ void processingDataCopy();

private:

	SimulationData* _data;
	Map<Cell> _cellMap;
	Map<Particle> _origParticleMap;

	int _startParticleIndex;
	int _endParticleIndex;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleProcessorOnCopyData::init(SimulationData & data)
{
	_data = &data;
	_cellMap.init(data.size, data.cellMap);
	_origParticleMap.init(data.size, data.particleMap);

	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particles.getNumEntries();
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, _startParticleIndex, _endParticleIndex);

	__syncthreads();
}

__inline__ __device__ void ParticleProcessorOnCopyData::processingDataCopy()
{
	for (int particleIndex = _startParticleIndex; particleIndex <= _endParticleIndex; ++particleIndex) {
		Particle *origParticle = &_data->particles.getEntireArray()[particleIndex];
		if (!origParticle->alive) {
			continue;
		}
		if (auto cell = _cellMap.get(origParticle->pos)) {
			if (cell->alive) {
				atomicAdd(&cell->energy, origParticle->energy);
				continue;
			}
		}
		Particle* newParticle = _data->particlesNew.getNewElement();
		*newParticle = *origParticle;
	}
}
