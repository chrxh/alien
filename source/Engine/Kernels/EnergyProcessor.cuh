#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"

namespace cg = cooperative_groups;

class EnergyProcessor
{
public:
    __inline__ __device__ static void updateGrid(SimulationData& data);
    __inline__ __device__ static void fillDensityGrid(SimulationData& data);
    __inline__ __device__ static void calcActiveSources(SimulationData& data);
    __inline__ __device__ static void moveParticlesFarFromSolids(SimulationData& data);
    __inline__ __device__ static void moveParticlesNearSolids(SimulationData& data);
    __inline__ __device__ static void splitHighEnergyParticles(SimulationData& data);
    __inline__ __device__ static void transformIntoFreeCells(SimulationData& data);

    __inline__ __device__ static void radiate(SimulationData& data, Object* cell, float energy);
    __inline__ __device__ static void createEnergyParticle(SimulationData& data, float2 pos, float2 vel, int color, float energy);
    __inline__ __device__ static void provideExternalEnergyForSources(SimulationData& data);

private:
    static auto constexpr SolidScanGroupSize = 8;
    using SolidScanGroup = cg::thread_block_tile<SolidScanGroupSize>;

    struct ConnectionHit
    {
        float fraction = NoConnectionHit;  // Fraction of the remaining displacement until the hit
        float2 normal = {0, 0};
        float2 velocity = {0, 0};
    };

    __inline__ __device__ static void calcPositionAndVelocityInSource(SimulationData& data, int sourceIndex, float2& pos, float2& vel);
    __inline__ __device__ static float takeExternalEnergy(SimulationData& data, float energy);
    __inline__ __device__ static float calcScanRadius(float solidSearchRadius, float2 const& displacement);
    __inline__ __device__ static void
    moveParticleAndBounceOffSolids(SimulationData& data, SolidScanGroup const& group, Energy* particle, float solidSearchRadius);
    __inline__ __device__ static bool findFirstConnectionHit(
        SimulationData& data,
        SolidScanGroup const& group,
        float2 const& pos,
        float2 const& vel,
        float timeLeft,
        float elapsedTime,
        float solidSearchRadius,
        ConnectionHit& hit);
    __inline__ __device__ static void
    updateFirstConnectionHit(SimulationData& data, Object* solid, float2 const& pos, float2 const& vel, float timeLeft, float elapsedTime, ConnectionHit& hit);
    __inline__ __device__ static void mergeOrAbsorb(SimulationData& data, Energy*& particle);
    __inline__ __device__ static void mergeParticleInto(Energy*& particle, Energy* target);
    __inline__ __device__ static void absorbIntoObject(SimulationData& data, Energy*& particle, Object* object);

    static auto constexpr MinEnergyPerSourceParticle = 10.0f;
    static auto constexpr MaxNumParticlesPerSource = 1000;
    static auto constexpr MaxBouncesPerTimestep = 3;
    static auto constexpr BounceClearance = 0.01f;
    static auto constexpr MaxSolidScanRadius = 8.0f;
    static auto constexpr NoConnectionHit = 2.0f;

    static_assert(2 * MaxSolidScanRadius + 1 <= OccupancyGrid::MaxAreaSize);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void EnergyProcessor::updateGrid(SimulationData& data)
{
    auto partition = calcBlockPartition(data.entities.energies.getNumOrigEntries());

    Energy** particlePointers = &data.entities.energies.at(partition.startIndex);
    data.energyParticleGrid.set_block(partition.numElements(), particlePointers);
}

__inline__ __device__ void EnergyProcessor::fillDensityGrid(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        data.preprocessedSimulationData.densityGrid.addParticle(data.entities.energies.at(index));
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

// Particles near solids are only collected here and moved by moveParticlesNearSolids
__inline__ __device__ void EnergyProcessor::moveParticlesFarFromSolids(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.energies.getNumOrigEntries());
    auto solidSearchRadius = data.solidGrid.calcSolidSearchRadius();
    auto timestepSize = cudaSimulationParameters.timestepSize.value;

    for (int particleIndex = partition.startIndex; particleIndex <= partition.endIndex; particleIndex += partition.step) {
        auto& particle = data.entities.energies.at(particleIndex);
        auto displacement = particle->vel * timestepSize;
        if (data.solidGrid.hasSolid(particle->pos + displacement / 2, calcScanRadius(solidSearchRadius, displacement))) {
            data.energyParticlesNearSolids.tryAddEntry(particleIndex);
        } else {
            particle->pos = particle->pos + displacement;
            data.world.correctPosition(particle->pos);
            mergeOrAbsorb(data, particle);
        }
    }
}

__inline__ __device__ void EnergyProcessor::moveParticlesNearSolids(SimulationData& data)
{
    auto group = cg::tiled_partition<SolidScanGroupSize>(cg::this_thread_block());
    auto groupIndex = toInt(blockIdx.x * blockDim.x + threadIdx.x) / SolidScanGroupSize;
    auto numGroups = toInt(gridDim.x * blockDim.x) / SolidScanGroupSize;
    auto solidSearchRadius = data.solidGrid.calcSolidSearchRadius();

    auto const& particlesNearSolids = data.energyParticlesNearSolids;
    for (int index = groupIndex; index < particlesNearSolids.getNumEntries(); index += numGroups) {
        auto& particle = data.entities.energies.at(particlesNearSolids.at(index));
        moveParticleAndBounceOffSolids(data, group, particle, solidSearchRadius);
        if (group.thread_rank() == 0) {
            data.world.correctPosition(particle->pos);
            mergeOrAbsorb(data, particle);
        }
    }
}

__inline__ __device__ float EnergyProcessor::calcScanRadius(float solidSearchRadius, float2 const& displacement)
{
    return min(solidSearchRadius + Math::length(displacement) / 2, MaxSolidScanRadius);
}

// All threads of the group compute the same movement, the first thread writes it. They start from the values read by the first thread,
// since other threads may change the particle meanwhile.
__inline__ __device__ void
EnergyProcessor::moveParticleAndBounceOffSolids(SimulationData& data, SolidScanGroup const& group, Energy* particle, float solidSearchRadius)
{
    auto readByFirstThread = [&](float2 const& value) { return float2{group.shfl(value.x, 0), group.shfl(value.y, 0)}; };
    auto pos = readByFirstThread(particle->pos);
    auto vel = readByFirstThread(particle->vel);
    auto timestepSize = cudaSimulationParameters.timestepSize.value;
    auto timeLeft = timestepSize;
    auto bounced = false;

    for (int i = 0; i < MaxBouncesPerTimestep; ++i) {
        ConnectionHit hit;
        if (!findFirstConnectionHit(data, group, pos, vel, timeLeft, timestepSize - timeLeft, solidSearchRadius, hit)) {
            pos = pos + vel * timeLeft;
            break;
        }
        pos = pos + vel * (timeLeft * hit.fraction) + hit.normal * BounceClearance;
        auto relativeVel = vel - hit.velocity;
        vel = relativeVel - hit.normal * (2 * Math::dot(relativeVel, hit.normal)) + hit.velocity;
        timeLeft *= 1.0f - hit.fraction;
        bounced = true;
    }

    if (group.thread_rank() == 0) {
        particle->pos = pos;
        if (bounced) {
            particle->vel = vel;
        }
    }
}

// The positions around the path are distributed among the threads of the group, the earliest hit of all threads is returned to each thread
__inline__ __device__ bool EnergyProcessor::findFirstConnectionHit(
    SimulationData& data,
    SolidScanGroup const& group,
    float2 const& pos,
    float2 const& vel,
    float timeLeft,
    float elapsedTime,
    float solidSearchRadius,
    ConnectionHit& hit)
{
    ConnectionHit threadHit;
    auto displacement = vel * timeLeft;
    auto scanRadius = calcScanRadius(solidSearchRadius, displacement);
    data.objectGrid.executeForEachSolidRecord(
        data.solidGrid, pos + displacement / 2, scanRadius, toInt(group.thread_rank()), toInt(group.size()), [&](LightObject const& record) {
            if (record.numConnections > 0) {
                updateFirstConnectionHit(data, record.self, pos, vel, timeLeft, elapsedTime, threadHit);
            }
        });

    hit.fraction = cg::reduce(group, threadHit.fraction, cg::less<float>());
    auto firstHitThread = __ffsll(static_cast<unsigned long long>(group.ballot(threadHit.fraction == hit.fraction))) - 1;
    hit.normal = {group.shfl(threadHit.normal.x, firstHitThread), group.shfl(threadHit.normal.y, firstHitThread)};
    hit.velocity = {group.shfl(threadHit.velocity.x, firstHitThread), group.shfl(threadHit.velocity.y, firstHitThread)};
    return hit.fraction != NoConnectionHit;
}

// Checks the connections from the solid to its connected solids and keeps the crossing if it comes before the given hit
__inline__ __device__ void EnergyProcessor::updateFirstConnectionHit(
    SimulationData& data,
    Object* solid,
    float2 const& pos,
    float2 const& vel,
    float timeLeft,
    float elapsedTime,
    ConnectionHit& hit)
{
    auto connectionStart = data.world.getCorrectedDirection(solid->pos - pos) + solid->vel * elapsedTime;
    for (int i = 0; i < solid->numConnections; ++i) {
        auto connectedObject = solid->connections[i].object;
        if (connectedObject->type != ObjectType_Solid) {
            continue;
        }
        auto connection = data.world.getCorrectedDirection(connectedObject->pos - solid->pos) + (connectedObject->vel - solid->vel) * elapsedTime;

        // The crossing is tested in the reference frame of the connection
        auto relativeDisplacement = (vel - (solid->vel + connectedObject->vel) / 2) * timeLeft;
        auto crossProduct = relativeDisplacement.x * connection.y - relativeDisplacement.y * connection.x;
        if (crossProduct == 0.0f) {
            continue;
        }
        auto fraction = (connectionStart.x * connection.y - connectionStart.y * connection.x) / crossProduct;
        if (fraction <= 0.0f || fraction > 1.0f || fraction >= hit.fraction) {
            continue;
        }
        auto connectionFraction = (connectionStart.x * relativeDisplacement.y - connectionStart.y * relativeDisplacement.x) / crossProduct;
        if (connectionFraction < 0.0f || connectionFraction > 1.0f) {
            continue;
        }

        auto connectionVel = solid->vel + (connectedObject->vel - solid->vel) * connectionFraction;
        auto normal = float2{-connection.y, connection.x} / Math::length(connection);
        if (crossProduct < 0.0f) {
            normal = normal * -1.0f;
        }
        if (Math::dot(vel - connectionVel, normal) >= 0.0f) {
            continue;
        }
        hit.fraction = fraction;
        hit.normal = normal;
        hit.velocity = connectionVel;
    }
}

__inline__ __device__ void EnergyProcessor::mergeOrAbsorb(SimulationData& data, Energy*& particle)
{
    auto otherParticle = data.energyParticleGrid.get(particle->pos);
    if (otherParticle && otherParticle != particle && Math::lengthSquared(particle->pos - otherParticle->pos) < 0.5) {
        mergeParticleInto(particle, otherParticle);
    } else if (auto object = data.objectGrid.getFirst(particle->pos + particle->vel)) {
        absorbIntoObject(data, particle, object);
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
    if (object->type == ObjectType_Fluid || object->type == ObjectType_Solid) {
        return;
    }
    auto isDigestor = object->type == ObjectType_Cell && object->typeData.cell.cellType == CellType_Digestor;
    if (object->isStatic() && !isDigestor) {
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
            data.world.correctPosition(otherPos);

            particle->pos -= velPerturbation / 5;
            data.world.correctPosition(particle->pos);

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
    data.world.correctPosition(particlePos);

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

    data.world.correctPosition(pos);

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
        data.world.correctPosition(pos);
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
                data.world.correctPosition(pos);
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
