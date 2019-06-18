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
	__inline__ __device__ void init_blockCall(SimulationData& data);

	__inline__ __device__ void processingDataCopy_blockCall();

private:

	SimulationData* _data;
	Map<Cell> _cellMap;
	Map<Particle> _origParticleMap;

    BlockData _particleBlock;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleProcessorOnCopyData::init_blockCall(SimulationData & data)
{
    _data = &data;
    _cellMap.init(data.size, data.cellMap);
    _origParticleMap.init(data.size, data.particleMap);

    _particleBlock = 
        calcPartition(data.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__inline__ __device__ void ParticleProcessorOnCopyData::processingDataCopy_blockCall()
{
	for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
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
