#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ParticleProcessor
{
public:
    __inline__ __device__ void updateMap(SimulationData& data);
    __inline__ __device__ void movement(SimulationData& data);
    __inline__ __device__ void collision(SimulationData& data);
    __inline__ __device__ void transformation(SimulationData& data, int numParticlePointers);

    /*
	__inline__ __device__ void init_system(SimulationData& data);

    __inline__ __device__ void processingMovement_system();
    __inline__ __device__ void updateMap_system();
    __inline__ __device__ void processingCollision_system();
    __inline__ __device__ void processingTransformation_system();
	__inline__ __device__ void processingDataCopy_system();

    __inline__ __device__ void repair_system();
*/
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
/*
__inline__ __device__ void ParticleProcessor::init_system(SimulationData & data)
{
    _data = &data;

    _particleBlock = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
}

__inline__ __device__ void ParticleProcessor::processingMovement_system()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        Particle* particle = _data->entities.particlePointers.at(particleIndex);
        particle->absPos = particle->absPos + particle->vel;
        _data->particleMap.mapPosCorrection(particle->absPos);
    }
}

__inline__ __device__ void ParticleProcessor::updateMap_system()
{
    auto const particleBlock = calcPartition(_data->entities.particlePointers.getNumEntries(), blockIdx.x, gridDim.x);

    Particle** particlePointers = &_data->entities.particlePointers.at(particleBlock.startIndex);
    _data->particleMap.set_block(particleBlock.numElements(), particlePointers);
}

__inline__ __device__ void ParticleProcessor::processingCollision_system()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        Particle* particle = _data->entities.particlePointers.at(particleIndex);
        Particle* otherParticle = _data->particleMap.get(particle->absPos);
        if (otherParticle && otherParticle != particle) {
            if (1 == particle->alive && 1 == otherParticle->alive) {

                SystemDoubleLock lock;
                lock.init(&particle->locked, &otherParticle->locked);
                lock.tryLock();
                if (!lock.isLocked()) {
                    continue;
                }

                auto const particleEnergy = particle->getEnergy_safe();
                float factor1 = particleEnergy / (particleEnergy + otherParticle->getEnergy_safe());
                float factor2 = 1.0f - factor1;
                particle->vel = particle->vel * factor1 + otherParticle->vel * factor2;
                particle->changeEnergy(otherParticle->getEnergy_safe());
                otherParticle->setEnergy_safe(0);
                atomicExch(&otherParticle->alive, 0);

                lock.releaseLock();
            }
        }
    }
}

__inline__ __device__ void ParticleProcessor::processingTransformation_system()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        if (_data->numberGen.random() < cudaSimulationParameters.cellTransformationProb) {
            auto& particle = _data->entities.particlePointers.getArrayForDevice()[particleIndex];
            auto innerEnergy = particle->getEnergy_safe()- Physics::linearKineticEnergy(1.0f, particle->vel);
            if (innerEnergy >= cudaSimulationParameters.cellMinEnergy) {
                EntityFactory factory;
                factory.init(_data);
                auto const cluster = factory.createClusterWithRandomCell(innerEnergy, particle->absPos, particle->vel);
                cluster->cellPointers[0]->metadata.color = particle->metadata.color;
                atomicExch(&particle->alive, 0);
            }
        }
    }
}

__inline__ __device__ void ParticleProcessor::processingDataCopy_system()
{
	for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
		auto& particle = _data->entities.particlePointers.at(particleIndex);
		if (0 == particle->alive) {
            particle = nullptr;
            continue;
		}
        if (auto cell = _data->cellMap.get(particle->absPos)) {
			if (1 == cell->alive) {
                cell->changeEnergy_safe(particle->getEnergy_safe());
                particle = nullptr;
			}
		}
	}
}

__inline__ __device__ void ParticleProcessor::repair_system()
{
    for (int particleIndex = _particleBlock.startIndex; particleIndex <= _particleBlock.endIndex; ++particleIndex) {
        auto& particle = _data->entities.particlePointers.at(particleIndex);
        particle->repair();
    }
}
*/

__inline__ __device__ void ParticleProcessor::updateMap(SimulationData& data)
{
    auto partition = calcPartition(data.entities.particlePointers.getNumEntries(), blockIdx.x, gridDim.x);

    Particle** particlePointers = &data.entities.particlePointers.at(partition.startIndex);
    data.particleMap.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void ParticleProcessor::movement(SimulationData& data)
{
    auto partition = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.entities.particlePointers.at(particleIndex);
        particle->absPos = particle->absPos + particle->vel;
        data.particleMap.mapPosCorrection(particle->absPos);
    }
}

__inline__ __device__ void ParticleProcessor::collision(SimulationData& data)
{
    auto partition = calcPartition(
        data.entities.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.entities.particlePointers.at(particleIndex);
        auto otherParticle = data.particleMap.get(particle->absPos);
        if (otherParticle && otherParticle != particle
            && Math::lengthSquared(particle->absPos - otherParticle->absPos) < 0.5) {

            SystemDoubleLock lock;
            lock.init(&particle->locked, &otherParticle->locked);
            if (lock.tryLock()) {
                __threadfence();

                if (particle->energy > FP_PRECISION && otherParticle->energy > FP_PRECISION) {
                    auto factor1 = particle->energy / (particle->energy + otherParticle->energy);
                    otherParticle->vel = particle->vel * factor1 + otherParticle->vel * (1.0f - factor1);
                    otherParticle->energy += particle->energy;
                    particle->energy = 0;
                    particle = nullptr;
                }

                __threadfence();
                lock.releaseLock();
            }
        } else {
            if (auto cell = data.cellMap.getFirst(particle->absPos)) {
                if (particle->tryLock()) {
                    __threadfence();
                    
                    atomicAdd(&cell->energy, particle->energy);
                    particle->energy = 0;

                    __threadfence();
                    particle->releaseLock();

                    particle = nullptr;
                }
            }
        }
    }
}

__inline__ __device__ void ParticleProcessor::transformation(SimulationData& data, int numParticlePointers)
{
    auto partition = calcPartition(numParticlePointers, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.entities.particlePointers.at(particleIndex)) {
            if (particle->energy >= cudaSimulationParameters.cellMinEnergy) {
                EntityFactory factory;
                factory.init(&data);
                auto cell = factory.createRandomCell(particle->energy, particle->absPos, particle->vel);
                cell->metadata.color = particle->metadata.color;

                particle = nullptr;
            }
        }
    }
}
