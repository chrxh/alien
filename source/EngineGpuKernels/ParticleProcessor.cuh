#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"
#include "ConstantMemory.cuh"
#include "ObjectFactory.cuh"

class ParticleProcessor
{
public:
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void movement(SimulationData& data);
    __inline__ __device__ static void collision(SimulationData& data);
    __inline__ __device__ static void transformation(SimulationData& data);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ParticleProcessor::updateMap(SimulationData& data)
{
    auto partition = calcPartition(data.objects.particlePointers.getNumEntries(), blockIdx.x, gridDim.x);

    Particle** particlePointers = &data.objects.particlePointers.at(partition.startIndex);
    data.particleMap.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void ParticleProcessor::movement(SimulationData& data)
{
    auto partition = calcPartition(
        data.objects.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particlePointers.at(particleIndex);
        particle->absPos = particle->absPos + particle->vel;
        data.particleMap.correctPosition(particle->absPos);
    }
}

__inline__ __device__ void ParticleProcessor::collision(SimulationData& data)
{
    auto partition = calcPartition(
        data.objects.particlePointers.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particlePointers.at(particleIndex);
        auto otherParticle = data.particleMap.get(particle->absPos);
        if (otherParticle && otherParticle != particle
            && Math::lengthSquared(particle->absPos - otherParticle->absPos) < 0.5) {

            SystemDoubleLock lock;
            lock.init(&particle->locked, &otherParticle->locked);
            if (lock.tryLock()) {

                if (particle->energy > NEAR_ZERO && otherParticle->energy > NEAR_ZERO) {
                    auto factor1 = particle->energy / (particle->energy + otherParticle->energy);
                    otherParticle->vel = particle->vel * factor1 + otherParticle->vel * (1.0f - factor1);
                    otherParticle->energy += particle->energy;
                    particle->energy = 0;
                    particle = nullptr;
                }

                lock.releaseLock();
            }
        } else {
            if (auto cell = data.cellMap.getFirst(particle->absPos)) {
                if (cell->barrier) {
                    continue;
                }
                if (cudaSimulationParameters.radiationAbsorption[cell->color] < NEAR_ZERO) {
                    continue;
                }
                if (!cell->tryLock()) {
                    continue;
                }
                if (particle->tryLock()) {

                    auto energyToTransfer = particle->energy * cudaSimulationParameters.radiationAbsorption[cell->color];
                    if (particle->energy < 5) {
                        energyToTransfer = particle->energy;
                    }
                    cell->energy += energyToTransfer;
                    particle->energy -= energyToTransfer;
                    bool killParticle = particle->energy < NEAR_ZERO;
                        
                    particle->releaseLock();

                    if (killParticle) {
                        particle = nullptr;
                    }
                }
                cell->releaseLock();
            }
        }
    }
}

__inline__ __device__ void ParticleProcessor::transformation(SimulationData& data)
{
    if (!cudaSimulationParameters.particleTransformationAllowed) {
        return;
    }
    auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumOrigEntries());
    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.objects.particlePointers.at(particleIndex)) {
            
            if (particle->energy >= cudaSimulationParameters.cellNormalEnergy[particle->color] && cudaSimulationParameters.radiationAbsorption[particle->color] > NEAR_ZERO) {
                ObjectFactory factory;
                factory.init(&data);
                auto cell = factory.createRandomCell(particle->energy, particle->absPos, particle->vel);
                cell->color = particle->color;

                particle = nullptr;
            }
        }
    }
}
