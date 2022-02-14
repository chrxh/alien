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
    __inline__ __device__ void transformation(SimulationData& data);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

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
        data.particleMap.correctPosition(particle->absPos);
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
                if (!cell->tryLock()) {
                    continue;
                }
                if (particle->tryLock()) {
                    
                    atomicAdd(&cell->energy, particle->energy);
                    particle->energy = 0;

                    particle->releaseLock();

                    particle = nullptr;
                }
                cell->releaseLock();
            }
        }
    }
}

__inline__ __device__ void ParticleProcessor::transformation(SimulationData& data)
{
    auto const partition = calcAllThreadsPartition(data.entities.particlePointers.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.entities.particlePointers.at(particleIndex)) {
            
            auto cellMinEnergy = SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, particle->absPos);
            if (particle->energy >= cellMinEnergy) {
                EntityFactory factory;
                factory.init(&data);
                auto cell = factory.createRandomCell(particle->energy, particle->absPos, particle->vel);
                cell->metadata.color = particle->metadata.color;

                particle = nullptr;
            }
        }
    }
}
