#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"

class EnergyProcessor
{
public:
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void fillDensityMap(SimulationData& data);
    __inline__ __device__ static void calcActiveSources(SimulationData& data);
    __inline__ __device__ static void movement(SimulationData& data);
    __inline__ __device__ static void collision(SimulationData& data);
    __inline__ __device__ static void splitting(SimulationData& data);
    __inline__ __device__ static void transformation(SimulationData& data);

    __inline__ __device__ static void radiate(SimulationData& data, Object* cell, float energy);
    __inline__ __device__ static void createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy);

private:
    static auto constexpr MaxFusionEnergy = 5.0f;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EnergyProcessor::updateMap(SimulationData& data)
{
    auto partition = calcBlockPartition(data.entities.energies.getNumOrigEntries());

    Energy** particlePointers = &data.entities.energies.at(partition.startIndex);
    data.energyMap.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void EnergyProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        data.preprocessedSimulationData.densityMap.addParticle(data.entities.energies.at(index));
    }
}

__inline__ __device__ void EnergyProcessor::calcActiveSources(SimulationData& data)
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

__inline__ __device__ void EnergyProcessor::movement(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        particle->pos = particle->pos + particle->vel * cudaSimulationParameters.timestepSize.value;
        data.energyMap.correctPosition(particle->pos);
    }
}

__inline__ __device__ void EnergyProcessor::collision(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        auto otherParticle = data.energyMap.get(particle->pos);
        if (otherParticle && otherParticle != particle && Math::lengthSquared(particle->pos - otherParticle->pos) < 0.5) {

            SystemDoubleLock lock;
            lock.init(&particle->locked, &otherParticle->locked);
            if (lock.tryLock()) {

                if (particle->energy > NEAR_ZERO && otherParticle->energy > NEAR_ZERO) {
                    auto factor1 = particle->energy / (particle->energy + otherParticle->energy);
                    otherParticle->vel = particle->vel * factor1 + otherParticle->vel * (1.0f - factor1);
                    otherParticle->energy += particle->energy;
                    otherParticle->lastAbsorbedObject = nullptr;
                    particle->energy = 0;
                    particle = nullptr;
                }

                lock.releaseLock();
            }
        } else {
            if (auto object = data.objectMap.getFirst(particle->pos + particle->vel)) {
                if (object->type == ObjectType_Solid || object->type == ObjectType_Fluid) {
                    continue;
                }
                if (object->fixed) {
                    auto vr = particle->vel - object->vel;
                    auto r = data.objectMap.getCorrectedDirection(particle->pos - object->pos);
                    auto dot_vr_r = Math::dot(vr, r);
                    if (dot_vr_r < 0) {
                        auto truncated_r_squared = max(0.1f, Math::lengthSquared(r));
                        particle->vel = vr - r * 2 * dot_vr_r / truncated_r_squared + object->vel;
                    }
                } else {
                    if (particle->lastAbsorbedObject == object) {
                        continue;
                    }
                    auto radiationAbsorption = ParameterCalculator::calcParameter(cudaSimulationParameters.radiationAbsorption, data, object->pos, object->color);

                    if (radiationAbsorption < NEAR_ZERO) {
                        continue;
                    }
                    if (!object->tryLock()) {
                        continue;
                    }
                    if (particle->tryLock()) {

                        auto energyToTransfer = particle->energy * radiationAbsorption;
                        if (cudaSimulationParameters.advancedAbsorptionControlToggle.value) {
                            energyToTransfer *=
                                max(0.0f, 1.0f - Math::length(object->vel) * cudaSimulationParameters.radiationAbsorptionHighVelocityPenalty.value[object->color]);

                            auto radiationAbsorptionLowVelocityPenalty = ParameterCalculator::calcParameter(
                                cudaSimulationParameters.radiationAbsorptionLowVelocityPenalty, data, object->pos, object->color);
                            energyToTransfer *= 1.0f - radiationAbsorptionLowVelocityPenalty / powf(1.0f + Math::length(object->vel), 10.0f);
                            energyToTransfer *= powf(
                                toFloat(object->numConnections + 1) / 7.0f, cudaSimulationParameters.radiationAbsorptionLowConnectionPenalty.value[object->color]);

                            //auto radiationAbsorptionLowNumCellsPenalty = ParameterCalculator::calcParameter(
                            //    cudaSimulationParameters.radiationAbsorptionLowNumCellsPenalty, data, object->pos, object->color);
                            //energyToTransfer *= 1.0f - radiationAbsorptionLowNumCellsPenalty / powf(1.0f + object->numObjects, 0.1f);
                        }

                        if (particle->energy < 0.01f /* && energyToTransfer > 0.1f*/) {
                            energyToTransfer = particle->energy;
                        }
                        if (object->type == ObjectType_Cell) {
                            object->typeData.cell.rawEnergy += energyToTransfer;
                        } else {
                            object->typeData.freeCell.energy += energyToTransfer;
                        }
                        particle->energy -= energyToTransfer;
                        bool killParticle = particle->energy < NEAR_ZERO;

                        particle->releaseLock();

                        if (killParticle) {
                            particle = nullptr;
                        } else {
                            particle->lastAbsorbedObject = object;
                        }
                    }
                    object->releaseLock();
                }
            }
        }
    }
}

__inline__ __device__ void EnergyProcessor::splitting(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        if (particle == nullptr) {
            continue;
        }
        if (data.primaryNumberGen.random() >= 0.01f) {
            continue;
        }

        if (particle->energy > cudaSimulationParameters.particleSplitEnergy.value[particle->color]) {
            particle->energy *= 0.5f;
            auto velPerturbation = Math::unitVectorOfAngle(data.primaryNumberGen.random() * 360);

            float2 otherPos = particle->pos + velPerturbation / 5;
            data.energyMap.correctPosition(otherPos);

            particle->pos -= velPerturbation / 5;
            data.energyMap.correctPosition(particle->pos);

            velPerturbation *= cudaSimulationParameters.radiationVelocityPerturbation / (particle->energy + 1.0f);
            float2 otherVel = particle->vel + velPerturbation;

            particle->vel -= velPerturbation;
            EntityFactory factory;
            factory.init(&data);
            factory.createEnergy(particle->energy, otherPos, otherVel, particle->color);
        }
    }
}

__inline__ __device__ void EnergyProcessor::transformation(SimulationData& data)
{
    if (!cudaSimulationParameters.particleTransformationAllowed.value) {
        return;
    }
    auto const partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());
    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        if (auto& particle = data.entities.energies.at(particleIndex)) {

            if (particle->energy >= cudaSimulationParameters.normalCellEnergy.value[particle->color]) {
                EntityFactory factory;
                factory.init(&data);
                auto object = factory.createFreeCell(particle->energy, particle->pos, particle->vel);
                object->color = particle->color;

                particle = nullptr;
            }
        }
    }
}

__inline__ __device__ void EnergyProcessor::radiate(SimulationData& data, Object* cell, float energy)
{
    auto const cellEnergy = atomicAdd(&cell->typeData.cell.usableEnergy, 0);

    auto const radiationEnergy = min(cellEnergy, energy);
    auto origEnergy = atomicAdd(&cell->typeData.cell.usableEnergy, -radiationEnergy);
    if (origEnergy < 1.0f) {
        atomicAdd(&cell->typeData.cell.usableEnergy, radiationEnergy);  //revert
        return;
    }

    float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
        + float2{
            (data.primaryNumberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
            (data.primaryNumberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};
    float2 particlePos = cell->pos + Math::getNormalized(particleVel) * 1.5f - particleVel;
    data.objectMap.correctPosition(particlePos);

    EnergyProcessor::createEnergyParticle(data, particlePos, particleVel, cell->color, radiationEnergy);
}

__inline__ __device__ void EnergyProcessor::createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy)
{
    auto numActiveSources = data.preprocessedSimulationData.activeRadiationSources.getNumActiveSources();
    if (numActiveSources > 0) {

        auto sumActiveRatios = 0.0f;
        for (int i = 0; i < numActiveSources; ++i) {
            auto index = data.preprocessedSimulationData.activeRadiationSources.getActiveSource(i);
            sumActiveRatios += cudaSimulationParameters.sourceRelativeStrength.sourceValues[index].value;
        }
        if (sumActiveRatios > 0) {
            auto randomRatioValue = data.primaryNumberGen.random(1.0f);
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
                        delta.x = data.primaryNumberGen.random() * radius * 2 - radius;
                        delta.y = data.primaryNumberGen.random() * radius * 2 - radius;
                        if (Math::length(delta) <= radius) {
                            break;
                        }
                    }
                    pos += delta;
                    if (cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].enabled) {
                        vel = Math::unitVectorOfAngle(cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].value)
                            * data.primaryNumberGen.random(0.5f, 1.0f);
                    } else {
                        vel = Math::getNormalized(delta) * data.primaryNumberGen.random(0.5f, 1.0f);
                    }
                }
                if (cudaSimulationParameters.sourceShapeType.sourceValues[sourceIndex] == SourceShapeType_Rectangular) {
                    auto const& rect = cudaSimulationParameters.sourceRectangularRect.sourceValues[sourceIndex];
                    float2 delta;
                    delta.x = data.primaryNumberGen.random() * rect.x - rect.x / 2;
                    delta.y = data.primaryNumberGen.random() * rect.y - rect.y / 2;
                    pos += delta;
                    if (cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].enabled) {
                        vel = Math::unitVectorOfAngle(cudaSimulationParameters.sourceRadiationAngle.sourceValues[sourceIndex].value)
                            * data.primaryNumberGen.random(0.5f, 1.0f);
                    } else {
                        auto roundSize = min(rect.x, rect.y) / 2;
                        float2 corner1{-rect.x / 2, -rect.y / 2};
                        float2 corner2{rect.x / 2, -rect.y / 2};
                        float2 corner3{-rect.x / 2, rect.y / 2};
                        float2 corner4{rect.x / 2, rect.y / 2};
                        if (Math::lengthMax(corner1 - delta) <= roundSize) {
                            vel = Math::getNormalized(delta - (corner1 + float2{roundSize, roundSize}));
                        } else if (Math::lengthMax(corner2 - delta) <= roundSize) {
                            vel = Math::getNormalized(delta - (corner2 + float2{-roundSize, roundSize}));
                        } else if (Math::lengthMax(corner3 - delta) <= roundSize) {
                            vel = Math::getNormalized(delta - (corner3 + float2{roundSize, -roundSize}));
                        } else if (Math::lengthMax(corner4 - delta) <= roundSize) {
                            vel = Math::getNormalized(delta - (corner4 + float2{-roundSize, -roundSize}));
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
                        vel = vel * data.primaryNumberGen.random(0.5f, 1.0f);
                    }
                }
            }
        }
    }

    data.objectMap.correctPosition(pos);

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

    auto particleEnergy = energy * (1.0f - externalEnergyBackflowFactor);
    if (particleEnergy > NEAR_ZERO) {
        EntityFactory factory;
        factory.init(&data);
        data.objectMap.correctPosition(pos);
        factory.createEnergy(particleEnergy, pos, vel, color);
    }
}
