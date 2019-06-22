#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "ModelBasic/ElementaryTypes.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"

class ParticleProcessorOnOrigData
{
public:
    __inline__ __device__ void init_blockCall(SimulationData& data);

    __inline__ __device__ void processingMovement_blockCall();
    __inline__ __device__ void processingCollision_blockCall();
    __inline__ __device__ void processingTransformation_blockCall();

private:
    SimulationData* _data;
    Map<Particle> _origParticleMap;
    BlockData _particleBlock;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ParticleProcessorOnOrigData::init_blockCall(SimulationData & data)
{
    _data = &data;
    _origParticleMap.init(data.size, data.particleMap);

    _particleBlock = calcPartition(data.entities.particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__inline__ __device__ void ParticleProcessorOnOrigData::processingMovement_blockCall()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        Particle* particle = &_data->entities.particles.getEntireArray()[particleIndex];
        particle->pos = Math::add(particle->pos, particle->vel);
        _origParticleMap.mapPosCorrection(particle->pos);
        _origParticleMap.set(particle->pos, particle);
    }
}

__inline__ __device__ void ParticleProcessorOnOrigData::processingCollision_blockCall()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        Particle* particle = &_data->entities.particles.getEntireArray()[particleIndex];
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
                particle->vel = Math::add(Math::mul(particle->vel, factor1), Math::mul(otherParticle->vel, factor2));
                atomicAdd(&particle->energy, otherParticle->energy);
                atomicAdd(&otherParticle->energy, -otherParticle->energy);
                otherParticle->alive = false;

                lock.releaseLock();
            }
        }
    }
}

__inline__ __device__ void ParticleProcessorOnOrigData::processingTransformation_blockCall()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        Particle* particle = &_data->entities.particles.getEntireArray()[particleIndex];
        auto innerEnergy = particle->energy - Physics::linearKineticEnergy(cudaSimulationParameters.cellMass, particle->vel);
        if (innerEnergy >= cudaSimulationParameters.cellMinEnergy) {
            if (_data->numberGen.random() < cudaSimulationParameters.cellTransformationProb) {
                EntityFactory factory;
                factory.init(_data);
                factory.createClusterWithRandomCell(innerEnergy, particle->pos, particle->vel);
                particle->alive = false;
            }
        }
    }
}
