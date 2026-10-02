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
    __inline__ __device__ static void moveAndBounceOffWalls(SimulationData& data);
    __inline__ __device__ static void mergeOrAbsorb(SimulationData& data);
    __inline__ __device__ static void splitHighEnergyParticles(SimulationData& data);
    __inline__ __device__ static void transformIntoFreeCells(SimulationData& data);

    __inline__ __device__ static void radiate(SimulationData& data, Object* cell, float energy);
    __inline__ __device__ static void createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy);
    __inline__ __device__ static void provideExternalEnergyForSources(SimulationData& data);

private:
    struct WallHit
    {
        float fraction;
        float2 normal;
        float2 velocity;
    };

    __inline__ __device__ static void calcPositionAndVelocityInSource(SimulationData& data, int sourceIndex, float2& pos, float2& vel);
    __inline__ __device__ static float takeExternalEnergy(SimulationData& data, float energy);
    __inline__ __device__ static bool isWall(Object* object);
    __inline__ __device__ static float calcWallSearchRadius();
    __inline__ __device__ static void moveParticleAndBounceOffWalls(SimulationData& data, Energy* particle, float wallSearchRadius);
    __inline__ __device__ static bool
    findFirstWall(SimulationData& data, float2 const& pos, float2 const& vel, float timeLeft, float elapsedTime, float wallSearchRadius, WallHit& hit);
    __inline__ __device__ static void mergeParticleInto(Energy*& particle, Energy* target);
    __inline__ __device__ static void absorbIntoObject(SimulationData& data, Energy*& particle, Object* object);

    static auto constexpr MinEnergyPerSourceParticle = 10.0f;
    static auto constexpr MaxNumParticlesPerSource = 1000;
    static auto constexpr MaxWallBouncesPerTimestep = 3;
    static auto constexpr WallClearance = 0.01f;
    static auto constexpr WallSearchMargin = 0.5f;
    static auto constexpr MaxWallScanRadius = 8.0f;
    static auto constexpr MinWallCrossProduct = 1.0e-6f;
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

__inline__ __device__ void EnergyProcessor::moveAndBounceOffWalls(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());
    auto wallSearchRadius = calcWallSearchRadius();

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        moveParticleAndBounceOffWalls(data, particle, wallSearchRadius);
        data.energyMap.correctPosition(particle->pos);
    }
}

__inline__ __device__ float EnergyProcessor::calcWallSearchRadius()
{
    // A wall crossed by a particle always has an endpoint within half of its maximum length from the crossing point, plus the distance the wall moves
    auto maxBindingDistance = 0.0f;
    for (int color = 0; color < MAX_COLORS; ++color) {
        maxBindingDistance = max(maxBindingDistance, cudaSimulationParameters.maxBindingDistance.value[color]);
    }
    return maxBindingDistance / 2 + WallSearchMargin + cudaSimulationParameters.maxVelocity.value * cudaSimulationParameters.timestepSize.value;
}

__inline__ __device__ void EnergyProcessor::moveParticleAndBounceOffWalls(SimulationData& data, Energy* particle, float wallSearchRadius)
{
    auto pos = particle->pos;
    auto vel = particle->vel;
    auto timestepSize = cudaSimulationParameters.timestepSize.value;
    auto timeLeft = timestepSize;
    auto bounced = false;

    for (int i = 0; i < MaxWallBouncesPerTimestep; ++i) {
        WallHit hit;
        if (!findFirstWall(data, pos, vel, timeLeft, timestepSize - timeLeft, wallSearchRadius, hit)) {
            pos = pos + vel * timeLeft;
            break;
        }
        pos = pos + vel * (timeLeft * hit.fraction) + hit.normal * WallClearance;
        auto relativeVel = vel - hit.velocity;
        vel = relativeVel - hit.normal * (2 * Math::dot(relativeVel, hit.normal)) + hit.velocity;
        timeLeft *= 1.0f - hit.fraction;
        bounced = true;
    }

    particle->pos = pos;
    if (bounced) {
        particle->vel = vel;
    }
}

__inline__ __device__ bool EnergyProcessor::findFirstWall(
    SimulationData& data,
    float2 const& pos,
    float2 const& vel,
    float timeLeft,
    float elapsedTime,
    float wallSearchRadius,
    WallHit& hit)
{
    hit.fraction = 2.0f;
    auto displacement = vel * timeLeft;
    auto scanRadius = min(wallSearchRadius + Math::length(displacement) / 2, MaxWallScanRadius);
    data.objectMap.executeForEachRecord(pos + displacement / 2, scanRadius, [&](LightObject const& record) {
        if (record.numConnections == 0 || (record.type != ObjectType_Solid && !record.isStatic())) {
            return;
        }
        auto object = record.self;
        if (!isWall(object)) {
            return;
        }

        auto wallStart = data.objectMap.getCorrectedDirection(object->pos - pos) + object->vel * elapsedTime;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if (!isWall(connectedObject)) {
                continue;
            }
            auto wall = data.objectMap.getCorrectedDirection(connectedObject->pos - object->pos) + (connectedObject->vel - object->vel) * elapsedTime;

            // The crossing is tested in the reference frame of the wall
            auto relativeDisplacement = (vel - (object->vel + connectedObject->vel) / 2) * timeLeft;
            auto crossProduct = relativeDisplacement.x * wall.y - relativeDisplacement.y * wall.x;
            if (abs(crossProduct) < MinWallCrossProduct) {
                continue;
            }
            auto fraction = (wallStart.x * wall.y - wallStart.y * wall.x) / crossProduct;
            if (fraction <= 0.0f || fraction > 1.0f || fraction >= hit.fraction) {
                continue;
            }
            auto wallFraction = (wallStart.x * relativeDisplacement.y - wallStart.y * relativeDisplacement.x) / crossProduct;
            if (wallFraction < 0.0f || wallFraction > 1.0f) {
                continue;
            }

            auto wallVel = object->vel + (connectedObject->vel - object->vel) * wallFraction;
            auto normal = float2{-wall.y, wall.x} / Math::length(wall);
            if (crossProduct < 0.0f) {
                normal = normal * -1.0f;
            }
            if (Math::dot(vel - wallVel, normal) >= 0.0f) {
                continue;
            }
            hit.fraction = fraction;
            hit.normal = normal;
            hit.velocity = wallVel;
        }
    });
    return hit.fraction <= 1.0f;
}

__inline__ __device__ bool EnergyProcessor::isWall(Object* object)
{
    if (object->type == ObjectType_Solid) {
        return true;
    }
    if (object->isStatic()) {
        return object->type != ObjectType_Cell || object->typeData.cell.cellType != CellType_Digestor;
    }
    return false;
}

__inline__ __device__ void EnergyProcessor::mergeOrAbsorb(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        auto otherParticle = data.energyMap.get(particle->pos);
        if (otherParticle && otherParticle != particle && Math::lengthSquared(particle->pos - otherParticle->pos) < 0.5) {
            mergeParticleInto(particle, otherParticle);
        } else if (auto object = data.objectMap.getFirst(particle->pos + particle->vel)) {
            absorbIntoObject(data, particle, object);
        }
    }
}

__inline__ __device__ void EnergyProcessor::mergeParticleInto(Energy*& particle, Energy* target)
{
    SystemDoubleLock lock;
    lock.init(&particle->locked, &target->locked);
    if (lock.tryLock()) {

        if (particle->energy > NEAR_ZERO && target->energy > NEAR_ZERO) {
            auto factor1 = particle->energy / (particle->energy + target->energy);
            target->vel = particle->vel * factor1 + target->vel * (1.0f - factor1);
            target->energy += particle->energy;
            target->lastAbsorbedObject = nullptr;
            particle->energy = 0;
            particle = nullptr;
        }

        lock.releaseLock();
    }
}

__inline__ __device__ void EnergyProcessor::absorbIntoObject(SimulationData& data, Energy*& particle, Object* object)
{
    if (object->type == ObjectType_Fluid || isWall(object)) {
        return;
    }
    if (particle->lastAbsorbedObject == object) {
        return;
    }
    if (object->type == ObjectType_Cell && object->typeData.cell.cellState == CellState_UnderConstruction) {
        return;
    }
    auto radiationAbsorption = ParameterCalculator::calcParameter(cudaSimulationParameters.radiationAbsorption, data, object->pos, object->color);

    if (radiationAbsorption < NEAR_ZERO) {
        return;
    }
    if (!object->tryLock()) {
        return;
    }
    if (particle->tryLock()) {

        auto energyToTransfer = particle->energy * radiationAbsorption;
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

__inline__ __device__ void EnergyProcessor::splitHighEnergyParticles(SimulationData& data)
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

__inline__ __device__ void EnergyProcessor::transformIntoFreeCells(SimulationData& data)
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
        atomicAdd(&cell->typeData.cell.usableEnergy, radiationEnergy);  // Revert
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
                calcPositionAndVelocityInSource(data, sourceIndex, pos, vel);
            }
        }
    }

    data.objectMap.correctPosition(pos);

    auto externalEnergyBackflowFactor = 0.0f;
    if (cudaSimulationParameters.externalEnergyBackflowFactor.value[color] > 0) {
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

__inline__ __device__ void EnergyProcessor::provideExternalEnergyForSources(SimulationData& data)
{
    auto totalInflow = 0.0f;
    for (int color = 0; color < MAX_COLORS; ++color) {
        totalInflow += cudaSimulationParameters.externalEnergyInflowForSources.value[color];
    }
    if (totalInflow < NEAR_ZERO) {
        return;
    }

    EntityFactory factory;
    factory.init(&data);

    auto numActiveSources = data.preprocessedSimulationData.activeRadiationSources.getNumActiveSources();
    auto const partition = calcSystemThreadPartition(numActiveSources);
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto sourceIndex = data.preprocessedSimulationData.activeRadiationSources.getActiveSource(index);
        auto relativeStrength = cudaSimulationParameters.sourceRelativeStrength.sourceValues[sourceIndex].value;

        for (int color = 0; color < MAX_COLORS; ++color) {
            auto requestedEnergy = cudaSimulationParameters.externalEnergyInflowForSources.value[color] * relativeStrength;
            if (requestedEnergy < NEAR_ZERO) {
                continue;
            }
            auto energy = takeExternalEnergy(data, requestedEnergy);
            if (energy < NEAR_ZERO) {
                continue;
            }

            auto numParticles = max(1, min(MaxNumParticlesPerSource, toInt(energy / MinEnergyPerSourceParticle)));
            auto particleEnergy = energy / toFloat(numParticles);
            for (int i = 0; i < numParticles; ++i) {
                float2 pos{0, 0};
                float2 vel{0, 0};
                calcPositionAndVelocityInSource(data, sourceIndex, pos, vel);
                data.objectMap.correctPosition(pos);
                factory.createEnergy(particleEnergy, pos, vel, color);
            }
        }
    }
}

__inline__ __device__ float EnergyProcessor::takeExternalEnergy(SimulationData& data, float energy)
{
    if (*data.externalEnergy == Infinity<float>::value) {
        return energy;
    }
    auto availableEnergy = atomicAdd(data.externalEnergy, -toDouble(energy));
    if (availableEnergy >= toDouble(energy)) {
        return energy;
    }
    auto grantedEnergy = max(0.0, availableEnergy);
    atomicAdd(data.externalEnergy, toDouble(energy) - grantedEnergy);  // Return the energy that is not available
    return toFloat(grantedEnergy);
}

__inline__ __device__ void EnergyProcessor::calcPositionAndVelocityInSource(SimulationData& data, int sourceIndex, float2& pos, float2& vel)
{
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
