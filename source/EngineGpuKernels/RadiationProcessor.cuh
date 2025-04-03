#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "Map.cuh"
#include "ConstantMemory.cuh"
#include "ObjectFactory.cuh"
#include "ZoneCalculator.cuh"

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
    auto partition = calcPartition(data.objects.particlePointers.getNumOrigEntries(), blockIdx.x, gridDim.x);

    Particle** particlePointers = &data.objects.particlePointers.at(partition.startIndex);
    data.particleMap.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void RadiationProcessor::calcActiveSources(SimulationData& data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int activeSourceIndex = 0;
        for (int i = 0; i < cudaSimulationParameters.numZones.value; ++i) {
            auto sourceActive = !ZoneCalculator::calcParameter(
                &SimulationParametersZoneValues::radiationDisableSources,
                &SimulationParametersZoneEnabledValues::radiationDisableSources,
                data,
                {cudaSimulationParameters.radiationSource[i].posX, cudaSimulationParameters.radiationSource[i].posY});
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
    auto partition = calcAllThreadsPartition(data.objects.particlePointers.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particlePointers.at(particleIndex);
        particle->absPos = particle->absPos + particle->vel * cudaSimulationParameters.timestepSize.value;
        data.particleMap.correctPosition(particle->absPos);
    }
}

__inline__ __device__ void RadiationProcessor::collision(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.objects.particlePointers.getNumOrigEntries());

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
                    otherParticle->lastAbsorbedCell = nullptr;
                    particle->energy = 0;
                    particle = nullptr;
                }

                lock.releaseLock();
            }
        } else {
            if (auto cell = data.cellMap.getFirst(particle->absPos + particle->vel)) {
                if (cell->barrier) {
                    auto vr = particle->vel - cell->vel;
                    auto r = data.cellMap.getCorrectedDirection(particle->absPos - cell->pos);
                    auto dot_vr_r = Math::dot(vr, r);
                    if (dot_vr_r < 0) {
                        auto truncated_r_squared = max(0.1f, Math::lengthSquared(r));
                        particle->vel = vr - r * 2 * dot_vr_r / truncated_r_squared + cell->vel;
                    }
                } else {
                    if (particle->lastAbsorbedCell == cell) {
                        continue;
                    }
                    auto radiationAbsorption = ZoneCalculator::calcParameterNew(cudaSimulationParameters.radiationAbsorption, data, cell->pos, cell->color);

                    if (radiationAbsorption < NEAR_ZERO) {
                        continue;
                    }
                    if (!cell->tryLock()) {
                        continue;
                    }
                    if (particle->tryLock()) {

                        auto energyToTransfer = particle->energy * radiationAbsorption;
                        if (cudaSimulationParameters.expertToggles.advancedAbsorptionControl) {
                            energyToTransfer *=
                                max(0.0f, 1.0f - Math::length(cell->vel) * cudaSimulationParameters.radiationAbsorptionHighVelocityPenalty[cell->color]);

                            auto radiationAbsorptionLowVelocityPenalty = ZoneCalculator::calcParameter(
                                &SimulationParametersZoneValues::radiationAbsorptionLowVelocityPenalty,
                                &SimulationParametersZoneEnabledValues::radiationAbsorptionLowVelocityPenalty,
                                data,
                                cell->pos,
                                cell->color);
                            energyToTransfer *= 1.0f - radiationAbsorptionLowVelocityPenalty / powf(1.0f + Math::length(cell->vel), 10.0f);
                            energyToTransfer *=
                                powf(toFloat(cell->numConnections + 1) / 7.0f, cudaSimulationParameters.radiationAbsorptionLowConnectionPenalty[cell->color]);

                            auto radiationAbsorptionLowGenomeComplexityPenalty = ZoneCalculator::calcParameter(
                                &SimulationParametersZoneValues::radiationAbsorptionLowGenomeComplexityPenalty,
                                &SimulationParametersZoneEnabledValues::radiationAbsorptionLowGenomeComplexityPenalty,
                                data,
                                cell->pos,
                                cell->color);
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
    auto partition = calcAllThreadsPartition(data.objects.particlePointers.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        auto& particle = data.objects.particlePointers.at(particleIndex);
        if (particle == nullptr) {
            continue;
        }
        if (data.numberGen1.random() >= 0.01f) {
            continue;
        }

        if (particle->energy > cudaSimulationParameters.particleSplitEnergy.value[particle->color]) {
            particle->energy *= 0.5f;
            auto velPerturbation = Math::unitVectorOfAngle(data.numberGen1.random() * 360);

            float2 otherPos = particle->absPos + velPerturbation / 5;
            data.particleMap.correctPosition(otherPos);

            particle->absPos -= velPerturbation / 5;
            data.particleMap.correctPosition(particle->absPos);

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
    auto const partition = calcAllThreadsPartition(data.objects.particlePointers.getNumOrigEntries());
    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; ++particleIndex) {
        if (auto& particle = data.objects.particlePointers.at(particleIndex)) {
            
            if (particle->energy >= cudaSimulationParameters.normalCellEnergy[particle->color]) {
                ObjectFactory factory;
                factory.init(&data);
                auto cell = factory.createFreeCell(particle->energy, particle->absPos, particle->vel);
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
            sumActiveRatios += cudaSimulationParameters.radiationSource[index].strength;
        }
        if (sumActiveRatios > 0) {
            auto randomRatioValue = data.numberGen1.random(1.0f);
            sumActiveRatios = 0.0f;
            auto sourceIndex = 0;
            auto matchSource = false;
            for (int i = 0; i < numActiveSources; ++i) {
                sourceIndex = data.preprocessedSimulationData.activeRadiationSources.getActiveSource(i);
                sumActiveRatios += cudaSimulationParameters.radiationSource[sourceIndex].strength;
                if (randomRatioValue <= sumActiveRatios) {
                    matchSource = true;
                    break;
                }
            }
            if (matchSource) {

                pos.x = cudaSimulationParameters.radiationSource[sourceIndex].posX;
                pos.y = cudaSimulationParameters.radiationSource[sourceIndex].posY;

                auto const& source = cudaSimulationParameters.radiationSource[sourceIndex];
                if (source.shape.type == RadiationSourceShapeType_Circular) {
                    auto radius = max(1.0f, source.shape.alternatives.circularRadiationSource.radius);
                    float2 delta{0, 0};
                    for (int i = 0; i < 10; ++i) {
                        delta.x = data.numberGen1.random() * radius * 2 - radius;
                        delta.y = data.numberGen1.random() * radius * 2 - radius;
                        if (Math::length(delta) <= radius) {
                            break;
                        }
                    }
                    pos += delta;
                    if (source.useAngle) {
                        vel = Math::unitVectorOfAngle(source.angle) * data.numberGen1.random(0.5f, 1.0f);
                    } else {
                        vel = Math::normalized(delta) * data.numberGen1.random(0.5f, 1.0f);
                    }
                }
                if (source.shape.type == RadiationSourceShapeType_Rectangular) {
                    auto const& rectangle = source.shape.alternatives.rectangularRadiationSource;
                    float2 delta;
                    delta.x = data.numberGen1.random() * rectangle.width - rectangle.width / 2;
                    delta.y = data.numberGen1.random() * rectangle.height - rectangle.height / 2;
                    pos += delta;
                    if (source.useAngle) {
                        vel = Math::unitVectorOfAngle(source.angle) * data.numberGen1.random(0.5f, 1.0f);
                    } else {
                        auto roundSize = min(rectangle.width, rectangle.height) / 2;
                        float2 corner1{-rectangle.width / 2, -rectangle.height / 2};
                        float2 corner2{rectangle.width / 2, -rectangle.height / 2};
                        float2 corner3{-rectangle.width / 2, rectangle.height / 2};
                        float2 corner4{rectangle.width / 2, rectangle.height / 2};
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
                            auto dx1 = rectangle.width / 2 + delta.x;
                            auto dx2 = rectangle.width / 2 - delta.x;
                            auto dy1 = rectangle.height / 2 + delta.y;
                            auto dy2 = rectangle.height / 2 - delta.y;
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
    if (cudaSimulationParameters.expertToggles.externalEnergyControl && cudaSimulationParameters.externalEnergyBackflowFactor[color] > 0) {
        auto energyToAdd = toDouble(energy * cudaSimulationParameters.externalEnergyBackflowFactor[color]);
        auto origExternalEnergy = atomicAdd(data.externalEnergy, energyToAdd);
        if (origExternalEnergy + energyToAdd > cudaSimulationParameters.externalEnergyBackflowLimit) {
            atomicAdd(data.externalEnergy, -energyToAdd);
        } else {
            externalEnergyBackflowFactor = cudaSimulationParameters.externalEnergyBackflowFactor[color];
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
