#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "Map.cuh"
#include "ConstantMemory.cuh"
#include "ObjectFactory.cuh"
#include "ParameterCalculator.cuh"

class RadiationProcessor
{
public:
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void calcActiveSources(SimulationData& data);
    __inline__ __device__ static void movement(SimulationData& data);
    __inline__ __device__ static void collision(SimulationData& data);
    __inline__ __device__ static void splitting(SimulationData& data);
    __inline__ __device__ static void transformation(SimulationData& data);

    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell, float energy);
    __inline__ __device__ static void createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy);

private:
    static auto constexpr MaxFusionEnergy = 5.0f;
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void RadiationProcessor::updateMap(SimulationData& data)
{
    auto partition = calcPartition(data.objects.particles.getNumOrigEntries(), blockIdx.x, gridDim.x);

    Particle** particlePointers = &data.objects.particles.at(partition.startIndex);
    data.particleMap.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void RadiationProcessor::calcActiveSources(SimulationData& data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int activeSourceIndex = 0;
        for (int i = 0; i < cudaSimulationParameters.numSources; ++i) {
            auto sourceActive = !ParameterCalculator::isCoveredByLayers(
                data,
                {cudaSimulationParameters.sourcePosition.sourceValues[i].x, cudaSimulationParameters.sourcePosition.sourceValues[i].y},
                cudaSimulationParameters.disableRadiationSources);
            if (sourceActive) {
                data.preprocessedSimulationData.activeRadiationSources.setActiveSource(activeSourceIndex, i);
                ++activeSourceIndex;
            }
        }
        data.preprocessedSimulationData.activeRadiationSources.setNumActiveSources(activeSourceIndex);
    }
}

__inline__ __device__ void RadiationProcessor::movement(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.objects.particles.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particles.at(particleIndex);
        particle->pos = particle->pos + particle->vel * cudaSimulationParameters.timestepSize.value;
        data.particleMap.correctPosition(particle->pos);
    }
}

__inline__ __device__ void RadiationProcessor::collision(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.objects.particles.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particles.at(particleIndex);
        auto otherParticle = data.particleMap.get(particle->pos);
        if (otherParticle && otherParticle != particle
            && Math::lengthSquared(particle->pos - otherParticle->pos) < 0.5) {

            SystemDoubleLock lock;
            lock.init(&particle->locked, &otherParticle->locked);
            if (lock.tryLock()) {

                if (particle->energy > NEAR_ZERO && otherParticle->energy > NEAR_ZERO) {
                    auto factor1 = particle->energy / (particle->energy + otherParticle->energy);
                    otherParticle->vel = particle->vel * factor1 + otherParticle->vel * (1.0f - factor1);
                    otherParticle->energy += particle->energy;
                    otherParticle->lastAbsorbedCell = nullptr;
                    particle->energy = 0;
                    particle = nullptr;
                }

                lock.releaseLock();
            }
        } else {
            if (auto cell = data.cellMap.getFirst(particle->pos + particle->vel)) {
                if (cell->barrier) {
                    auto vr = particle->vel - cell->vel;
                    auto r = data.cellMap.getCorrectedDirection(particle->pos - cell->pos);
                    auto dot_vr_r = Math::dot(vr, r);
                    if (dot_vr_r < 0) {
                        auto truncated_r_squared = max(0.1f, Math::lengthSquared(r));
                        particle->vel = vr - r * 2 * dot_vr_r / truncated_r_squared + cell->vel;
                    }
                } else {
                    if (particle->lastAbsorbedCell == cell) {
                        continue;
                    }
                    auto radiationAbsorption = ParameterCalculator::calcParameter(cudaSimulationParameters.radiationAbsorption, data, cell->pos, cell->color);

                    if (radiationAbsorption < NEAR_ZERO) {
                        continue;
                    }
                    if (!cell->tryLock()) {
                        continue;
                    }
                    if (particle->tryLock()) {

                        auto energyToTransfer = particle->energy * radiationAbsorption;
                        if (cudaSimulationParameters.advancedAbsorptionControlToggle.value) {
                            energyToTransfer *=
                                max(0.0f, 1.0f - Math::length(cell->vel) * cudaSimulationParameters.radiationAbsorptionHighVelocityPenalty.value[cell->color]);

                            auto radiationAbsorptionLowVelocityPenalty =
                                ParameterCalculator::calcParameter(cudaSimulationParameters.radiationAbsorptionLowVelocityPenalty, data, cell->pos, cell->color);
                            energyToTransfer *= 1.0f - radiationAbsorptionLowVelocityPenalty / powf(1.0f + Math::length(cell->vel), 10.0f);
                            energyToTransfer *=
                                powf(toFloat(cell->numConnections + 1) / 7.0f, cudaSimulationParameters.radiationAbsorptionLowConnectionPenalty.value[cell->color]);

                            auto radiationAbsorptionLowGenomeComplexityPenalty = ParameterCalculator::calcParameter(
                                cudaSimulationParameters.radiationAbsorptionLowGenomeComplexityPenalty, data, cell->pos, cell->color);
                            energyToTransfer *= 1.0f - radiationAbsorptionLowGenomeComplexityPenalty / powf(1.0f + cell->genomeComplexity, 0.1f);
                        }

                        if (particle->energy < 0.01f/* && energyToTransfer > 0.1f*/) {
                            energyToTransfer = particle->energy;
                        }
                        cell->energy += energyToTransfer;
                        particle->energy -= energyToTransfer;
                        bool killParticle = particle->energy < NEAR_ZERO;

                        particle->releaseLock();

                        if (killParticle) {
                            particle = nullptr;
                        } else {
                            particle->lastAbsorbedCell = cell;
                        }
                    }
                    cell->releaseLock();
                }
            }
        }
    }
}

__inline__ __device__ void RadiationProcessor::splitting(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.objects.particles.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particles.at(particleIndex);
        if (particle == nullptr) {
            continue;
        }
        if (data.numberGen1.random() >= 0.01f) {
            continue;
        }

        if (particle->energy > cudaSimulationParameters.particleSplitEnergy.value[particle->color]) {
            particle->energy *= 0.5f;
            auto velPerturbation = Math::unitVectorOfAngle(data.numberGen1.random() * 360);

            float2 otherPos = particle->pos + velPerturbation / 5;
            data.particleMap.correctPosition(otherPos);

            particle->pos -= velPerturbation / 5;
            data.particleMap.correctPosition(particle->pos);

            velPerturbation *= cudaSimulationParameters.radiationVelocityPerturbation / (particle->energy + 1.0f);
            float2 otherVel = particle->vel + velPerturbation;

            particle->vel -= velPerturbation;
            ObjectFactory factory;
            factory.init(&data);
            factory.createParticle(particle->energy, otherPos, otherVel, particle->color);
        }
    }
}

__inline__ __device__ void RadiationProcessor::transformation(SimulationData& data)
{
    if (!cudaSimulationParameters.particleTransformationAllowed.value) {
        return;
    }
    auto const partition = calcAllThreadsPartition(data.objects.particles.getNumOrigEntries());
    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.objects.particles.at(particleIndex)) {
            
            if (particle->energy >= cudaSimulationParameters.normalCellEnergy.value[particle->color]) {
                ObjectFactory factory;
                factory.init(&data);
                auto cell = factory.createFreeCell(particle->energy, particle->pos, particle->vel);
                cell->color = particle->color;

                particle = nullptr;
            }
        }
    }
}

__inline__ __device__ void RadiationProcessor::radiate(SimulationData& data, Cell* cell, float energy)
{
    auto const cellEnergy = atomicAdd(&cell->energy, 0);

    auto const radiationEnergy = min(cellEnergy, energy);
    auto origEnergy = atomicAdd(&cell->energy, -radiationEnergy);
    if (origEnergy < 1.0f) {
        atomicAdd(&cell->energy, radiationEnergy);  //revert
        return;
    }

    float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
        + float2{
            (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
            (data.numberGen1.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
    float2 particlePos = cell->pos + Math::normalized(particleVel) * 1.5f - particleVel;
    data.cellMap.correctPosition(particlePos);

    RadiationProcessor::createEnergyParticle(data, particlePos, particleVel, cell->color, radiationEnergy);
}

__inline__ __device__ void RadiationProcessor::createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy)
{
    auto numActiveSources = data.preprocessedSimulationData.activeRadiationSources.getNumActiveSources();
    if (numActiveSources > 0) {

        auto sumActiveRatios = 0.0f;
        for (int i = 0; i < numActiveSources; ++i) {
            auto index = data.preprocessedSimulationData.activeRadiationSources.getActiveSource(i);
            sumActiveRatios += cudaSimulationParameters.sourceRelativeStrength.sourceValues[index].value;
        }
        if (sumActiveRatios > 0) {
            auto randomRatioValue = data.numberGen1.random(1.0f);
            sumActiveRatios = 0.0f;
            auto sourceIndex = 0;
            auto matchSource = false;
            for (int i = 0; i < numActiveSources; ++i) {
                sourceIndex = data.preprocessedSimulationData.activeRadiationSources.getActiveSource(i);
                sumActiveRatios += cudaSimulationParameters.sourceRelativeStrength.sourceValues[sourceIndex].value;
                if (randomRatioValue <= sumActiveRatios) {
                    matchSource = true;
                    break;
                }
            }
            if (matchSource) {

                pos.x = cudaSimulationParameters.sourcePosition.sourceValues[sourceIndex].x;
                pos.y = cudaSimulationParameters.sourcePosition.sourceValues[sourceIndex].y;

                if (cudaSimulationParameters.sourceShapeType.sourceValues[sourceIndex] == SourceShapeType_Circular) {
                    auto radius = max(1.0f, cudaSimulationParameters.sourceCircularRadius.sourceValues[sourceIndex]);
                    float2 delta{0, 0};
                    for (int i = 0; i < 10; ++i) {
                        delta.x = data.numberGen1.random() * radius * 2 - radius;
                        delta.y = data.numberGen1.random() * radius * 2 - radius;
                        if (Math::length(delta) <= radius) {
                            break;
                        }
                    }
                    pos += delta;
                    if (cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].enabled) {
                        vel = Math::unitVectorOfAngle(cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].value)
                            * data.numberGen1.random(0.5f, 1.0f);
                    } else {
                        vel = Math::normalized(delta) * data.numberGen1.random(0.5f, 1.0f);
                    }
                }
                if (cudaSimulationParameters.sourceShapeType.sourceValues[sourceIndex] == SourceShapeType_Rectangular) {
                    auto const& rect = cudaSimulationParameters.sourceRectangularRect.sourceValues[sourceIndex];
                    float2 delta;
                    delta.x = data.numberGen1.random() * rect.x - rect.x / 2;
                    delta.y = data.numberGen1.random() * rect.y - rect.y / 2;
                    pos += delta;
                    if (cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].enabled) {
                        vel = Math::unitVectorOfAngle(cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].value)
                            * data.numberGen1.random(0.5f, 1.0f);
                    } else {
                        auto roundSize = min(rect.x, rect.y) / 2;
                        float2 corner1{-rect.x / 2, -rect.y / 2};
                        float2 corner2{rect.x / 2, -rect.y / 2};
                        float2 corner3{-rect.x / 2, rect.y / 2};
                        float2 corner4{rect.x / 2, rect.y / 2};
                        if (Math::lengthMax(corner1 - delta) <= roundSize) {
                            vel = Math::normalized(delta - (corner1 + float2{roundSize, roundSize}));
                        } else if (Math::lengthMax(corner2 - delta) <= roundSize) {
                            vel = Math::normalized(delta - (corner2 + float2{-roundSize, roundSize}));
                        } else if (Math::lengthMax(corner3 - delta) <= roundSize) {
                            vel = Math::normalized(delta - (corner3 + float2{roundSize, -roundSize}));
                        } else if (Math::lengthMax(corner4 - delta) <= roundSize) {
                            vel = Math::normalized(delta - (corner4 + float2{-roundSize, -roundSize}));
                        } else {
                            vel.x = 0;
                            vel.y = 0;
                            auto dx1 = rect.x / 2 + delta.x;
                            auto dx2 = rect.x / 2 - delta.x;
                            auto dy1 = rect.y / 2 + delta.y;
                            auto dy2 = rect.y / 2 - delta.y;
                            if (dx1 <= dy1 && dx1 <= dy2 && delta.x <= 0) {
                                vel.x = -1;
                            }
                            if (dy1 <= dx1 && dy1 <= dx2 && delta.y <= 0) {
                                vel.y = -1;
                            }
                            if (dx2 <= dy1 && dx2 <= dy2 && delta.x > 0) {
                                vel.x = 1;
                            }
                            if (dy2 <= dx1 && dy2 <= dx2 && delta.y > 0) {
                                vel.y = 1;
                            }
                        }
                        vel = vel * data.numberGen1.random(0.5f, 1.0f);
                    }
                }
            }
        }
    }

    data.cellMap.correctPosition(pos);

    auto externalEnergyBackflowFactor = 0.0f;
    if (cudaSimulationParameters.externalEnergyControlToggle.value && cudaSimulationParameters.externalEnergyBackflowFactor.value[color] > 0) {
        auto energyToAdd = toDouble(energy * cudaSimulationParameters.externalEnergyBackflowFactor.value[color]);
        auto origExternalEnergy = atomicAdd(data.externalEnergy, energyToAdd);
        if (origExternalEnergy + energyToAdd > cudaSimulationParameters.externalEnergyBackflowLimit.value) {
            atomicAdd(data.externalEnergy, -energyToAdd);
        } else {
            externalEnergyBackflowFactor = cudaSimulationParameters.externalEnergyBackflowFactor.value[color];
        }
    }

    auto particleEnergy =
        energy * (1.0f - externalEnergyBackflowFactor);
    if (particleEnergy > NEAR_ZERO) {
        ObjectFactory factory;
        factory.init(&data);
        data.cellMap.correctPosition(pos);
        factory.createParticle(particleEnergy, pos, vel, color);
    }
}
