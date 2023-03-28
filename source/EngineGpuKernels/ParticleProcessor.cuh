#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "Map.cuh"
#include "ConstantMemory.cuh"
#include "ObjectFactory.cuh"
#include "SpotCalculator.cuh"

class ParticleProcessor
{
public:
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void movement(SimulationData& data);
    __inline__ __device__ static void collision(SimulationData& data);
    __inline__ __device__ static void transformation(SimulationData& data);
    __inline__ __device__ static float2 getRadiationPos(SimulationData& data, float2 const& cellPos);
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
                auto radiationAbsorption = SpotCalculator::calcParameter(
                    &SimulationParametersSpotValues::radiationAbsorption,
                    &SimulationParametersSpotActivatedValues::radiationAbsorption,
                    data,
                    cell->absPos,
                    cell->color);

                if (radiationAbsorption < NEAR_ZERO) {
                    continue;
                }
                if (!cell->tryLock()) {
                    continue;
                }
                if (particle->tryLock()) {

                    auto energyToTransfer = particle->energy * radiationAbsorption;
                    if (particle->energy < 1) {
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
            
            if (particle->energy >= cudaSimulationParameters.cellNormalEnergy[particle->color]) {
                ObjectFactory factory;
                factory.init(&data);
                auto cell = factory.createRandomCell(particle->energy, particle->absPos, particle->vel);
                cell->color = particle->color;

                particle = nullptr;
            }
        }
    }
}

__inline__ __device__ float2 ParticleProcessor::getRadiationPos(SimulationData& data, float2 const& cellPos)
{
    auto result = cellPos;
    if (cudaSimulationParameters.numParticleSources > 0) {
        auto sourceIndex = data.numberGen1.random(cudaSimulationParameters.numParticleSources - 1);
        result.x = cudaSimulationParameters.particleSources[sourceIndex].posX;
        result.y = cudaSimulationParameters.particleSources[sourceIndex].posY;
    }
    return result;
}
