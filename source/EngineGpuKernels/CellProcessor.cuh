#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/CellFunctionEnums.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "ObjectFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "CellConnectionProcessor.cuh"
#include "SpotCalculator.cuh"

class CellProcessor
{
public:
    __inline__ __device__ static void init(SimulationData& data);
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void clearDensityMap(SimulationData& data);
    __inline__ __device__ static void fillDensityMap(SimulationData& data);

    __inline__ __device__ static void calcPressure(SimulationData& data);
    __inline__ __device__ static void calcFluidForcesAndReconnectCells(SimulationData& data);

    __inline__ __device__ static void calcCollisionsAndReconnectCells(SimulationData& data);
    __inline__ __device__ static void checkForces(SimulationData& data);
    __inline__ __device__ static void applyForces(SimulationData& data);  //prerequisite: data from calcCollisionsAndReconnectCells

    __inline__ __device__ static void calcConnectionForces(SimulationData& data, bool considerAngles);
    __inline__ __device__ static void checkConnections(SimulationData& data);
    __inline__ __device__ static void verletPositionUpdate(SimulationData& data);
    __inline__ __device__ static void verletVelocityUpdate(SimulationData& data);

    __inline__ __device__ static void livingStateTransition(SimulationData& data);

    __inline__ __device__ static void applyInnerFriction(SimulationData& data);
    __inline__ __device__ static void applyFriction(SimulationData& data);

    __inline__ __device__ static void radiation(SimulationData& data);
    __inline__ __device__ static void decay(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellProcessor::init(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->activityFetched = 0;
        cell->shared1 = {0, 0};
        cell->nextCell = nullptr;
        cell->density = 0;
    }
}

__inline__ __device__ void CellProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcPartition(data.objects.cellPointers.getNumEntries(), blockIdx.x, gridDim.x);
    Cell** cellPointers = &data.objects.cellPointers.at(partition.startIndex);
    data.cellMap.set_block(partition.numElements(), cellPointers);
}

__inline__ __device__ void CellProcessor::clearDensityMap(SimulationData& data)
{
    data.preprocessedCellFunctionData.densityMap.clear();
}

__inline__ __device__ void CellProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        data.preprocessedCellFunctionData.densityMap.addCell(data.objects.cellPointers.at(index));
    }
}

namespace
{
    __inline__ __device__ float calcKernel(float q)
    {
        float result;
        if (q < 1) {
            result = 2.0f / 3.0f - q * q + 0.5f * q * q * q;
        } else if (q < 2) {
            result = 2.0f - q;
            result = result * result * result / 6;
        } else {
            result = 0;
        }
        result *= 3.0f / (2.0f * Const::PI);
        return result;
    }

    __inline__ __device__ float calcKernel_d(float q)
    {
        float result;
        if (q < 1) {
            result = -2 * q + 3.0f / 2.0f * q * q;
        } else if (q < 2) {
            result = -0.5f * (2.0f - q) * (2.0f - q);
        } else {
            result = 0;
        }
        result *= 3.0f / (2.0f * Const::PI);
        return result;
    }
}

__inline__ __device__ void CellProcessor::calcPressure(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const& smoothingLength = cudaSimulationParameters.motionData.fluidMotion.smoothingLength;

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        float p = 0;
        data.cellMap.executeForEach(cell->absPos, 2 * smoothingLength, cell->detached, [&](auto const& otherCell) {
            auto q = data.cellMap.getDistance(cell->absPos, otherCell->absPos);
            p += calcKernel(q / smoothingLength) / (smoothingLength * smoothingLength);
        });

        cell->density = p;
    }
}

__inline__ __device__ void CellProcessor::calcFluidForcesAndReconnectCells(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());
    auto const& smoothingLength = cudaSimulationParameters.motionData.fluidMotion.smoothingLength;

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        auto cellMaxBindingEnergy = SpotCalculator::calcParameter(
            &SimulationParametersSpotValues::cellMaxBindingEnergy, &SimulationParametersSpotActivatedValues::cellMaxBindingEnergy, data, cell->absPos);

        float2 F_pressure = {0, 0};
        float2 F_viscosity = {0, 0};
        float2 F_boundary = {0, 0};
        data.cellMap.executeForEach(
            cell->absPos, 2 * smoothingLength, cell->detached, [&](auto const& otherCell) {
            if (cell == otherCell) {
                return;
            }
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const& connectedCell = cell->connections[i].cell;
                if (connectedCell == otherCell) {
                    return;
                }
            }

            auto posDelta = cell->absPos - otherCell->absPos;
            data.cellMap.correctDirection(posDelta);
            auto distance = Math::length(posDelta);
            auto velDelta = cell->vel - otherCell->vel;

            //calc forces
            if (!otherCell->barrier) {

                //for simplicity pressure = density
                auto const& cellPressure = cell->density;
                auto const& otherCellPressure = otherCell->density;
                auto factor = (cellPressure / (cell->density * cell->density) + otherCellPressure / (otherCell->density * otherCell->density));

                if (abs(distance) < NEAR_ZERO) {
                    return;
                }
                float kernel_d = calcKernel_d(distance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);
                F_pressure = F_pressure - posDelta / distance * factor * kernel_d;
                F_viscosity = F_viscosity + velDelta / otherCell->density * distance * kernel_d / (distance * distance + 0.25f);
            } else {
                F_boundary = F_boundary + posDelta * Math::dot(velDelta, posDelta) / (-distance * distance - 0.5f) * 2;
            }

            //reconnecting
            if (distance < cudaSimulationParameters.cellMinDistance && cell->numConnections > 1 && !cell->barrier
                && cell->livingState != LivingState_UnderConstruction) {
                CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            }

            auto isApproaching = Math::dot(posDelta, velDelta) < 0;
            if (cell->numConnections < cell->maxConnections && otherCell->numConnections < otherCell->maxConnections
                && Math::length(velDelta) >= SpotCalculator::calcParameter(
                       &SimulationParametersSpotValues::cellFusionVelocity, &SimulationParametersSpotActivatedValues::cellFusionVelocity, data, cell->absPos)
                && isApproaching && cell->energy <= cellMaxBindingEnergy && otherCell->energy <= cellMaxBindingEnergy && !cell->barrier
                && !otherCell->barrier) {
                CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
            }
        });
        cell->shared1 = cell->shared1 + F_pressure * cudaSimulationParameters.motionData.fluidMotion.pressureStrength
            + F_viscosity * cudaSimulationParameters.motionData.fluidMotion.viscosityStrength + F_boundary;
    }
}

__inline__ __device__ void CellProcessor::calcCollisionsAndReconnectCells(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        data.cellMap.executeForEach(
            cell->absPos, cudaSimulationParameters.motionData.collisionMotion.cellMaxCollisionDistance, cell->detached, [&](auto const& otherCell) {
                if (otherCell == cell) {
                    return;
                }

                auto posDelta = cell->absPos - otherCell->absPos;
                data.cellMap.correctDirection(posDelta);

                auto distance = Math::length(posDelta);
                if (distance < cudaSimulationParameters.cellMinDistance && cell->numConnections > 1 && !cell->barrier
                    && cell->livingState != LivingState_UnderConstruction) {
                    CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
                }

                bool alreadyConnected = false;
                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const& connectedCell = cell->connections[i].cell;
                    if (connectedCell == otherCell) {
                        alreadyConnected = true;
                        break;
                    }
                }

                if (!alreadyConnected) {
                    auto velDelta = cell->vel - otherCell->vel;
                    auto isApproaching = Math::dot(posDelta, velDelta) < 0;
                    auto barrierFactor = cell->barrier ? 2 : 1;

                    if (Math::length(cell->vel) > 0.5f && isApproaching) {
                        auto distanceSquared = distance * distance + 0.25f;
                        auto force = posDelta * Math::dot(velDelta, posDelta) / (-2 * distanceSquared) * barrierFactor;
                        atomicAdd(&cell->shared1.x, force.x);
                        atomicAdd(&cell->shared1.y, force.y);
                        atomicAdd(&otherCell->shared1.x, -force.x);
                        atomicAdd(&otherCell->shared1.y, -force.y);
                    } else {
                        auto force = Math::normalized(posDelta)
                            * (cudaSimulationParameters.motionData.collisionMotion.cellMaxCollisionDistance - Math::length(posDelta))
                            * cudaSimulationParameters.motionData.collisionMotion.cellRepulsionStrength * barrierFactor;
                        atomicAdd(&cell->shared1.x, force.x);
                        atomicAdd(&cell->shared1.y, force.y);
                        atomicAdd(&otherCell->shared1.x, -force.x);
                        atomicAdd(&otherCell->shared1.y, -force.y);
                    }

                    auto cellMaxBindingEnergy = SpotCalculator::calcParameter(
                        &SimulationParametersSpotValues::cellMaxBindingEnergy,
                        &SimulationParametersSpotActivatedValues::cellMaxBindingEnergy,
                        data,
                        cell->absPos);

                    if (cell->numConnections < cell->maxConnections && otherCell->numConnections < otherCell->maxConnections
                        && Math::length(velDelta) >= SpotCalculator::calcParameter(
                               &SimulationParametersSpotValues::cellFusionVelocity,
                               &SimulationParametersSpotActivatedValues::cellFusionVelocity,
                               data,
                               cell->absPos)
                        && isApproaching && cell->energy <= cellMaxBindingEnergy && otherCell->energy <= cellMaxBindingEnergy && !cell->barrier
                        && !otherCell->barrier) {
                        CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                    }
                }
            });
    }
}

__inline__ __device__ void CellProcessor::checkForces(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (auto index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        if (Math::length(cell->shared1) > SpotCalculator::calcParameter(
                &SimulationParametersSpotValues::cellMaxForce, &SimulationParametersSpotActivatedValues::cellMaxForce, data, cell->absPos)) {
            if (data.numberGen1.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
                CellConnectionProcessor::scheduleDeleteCell(data, index);
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyForces(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition =
        calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        cell->vel = cell->vel + cell->shared1 * cudaSimulationParameters.timestepSize;
        if (Math::length(cell->vel) > cudaSimulationParameters.cellMaxVelocity) {
            cell->vel = Math::normalized(cell->vel) * cudaSimulationParameters.cellMaxVelocity;
        }
        cell->shared1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::calcConnectionForces(SimulationData& data, bool considerAngles)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (0 == cell->numConnections || cell->barrier) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = cell->connections[cell->numConnections - 1].cell->absPos - cell->absPos;
        data.cellMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = cell->stiffness * cell->stiffness;

        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            auto connectedCellStiffnessSquared = connectedCell->stiffness * connectedCell->stiffness;

            auto displacement = connectedCell->absPos - cell->absPos;
            data.cellMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::normalized(displacement) * deviation * (cellStiffnessSquared + connectedCellStiffnessSquared) / 4;

            if (considerAngles && cell->numConnections >= 2) {

                auto lastIndex = (i + cell->numConnections - 1) % cell->numConnections;
                auto lastConnectedCell = cell->connections[lastIndex].cell;

                //check if there is a triangular connection
                bool triangularConnection = false;
                for (int j = 0; j < connectedCell->numConnections; ++j) {
                    if (connectedCell->connections[j].cell == lastConnectedCell) {
                        triangularConnection = true;
                        break;
                    }
                }

                //angle forces in case of no triangular connections
                if (!triangularConnection) {
                    auto angle = Math::angleOfVector(displacement);
                    auto prevAngle = Math::angleOfVector(prevDisplacement);
                    auto actualAngleFromPrevious = Math::subtractAngle(angle, prevAngle);
                    auto referenceAngleFromPrevious = cell->connections[i].angleFromPrevious;

                    auto strength = abs(referenceAngleFromPrevious - actualAngleFromPrevious) / 2000 * cellStiffnessSquared;

                    auto force1 = Math::normalized(displacement) / max(Math::length(displacement), cudaSimulationParameters.cellMinDistance) * strength;
                    Math::rotateQuarterClockwise(force1);

                    auto force2 =
                        Math::normalized(prevDisplacement) / max(Math::length(prevDisplacement), cudaSimulationParameters.cellMinDistance) * strength;
                    Math::rotateQuarterCounterClockwise(force2);

                    if (abs(referenceAngleFromPrevious - actualAngleFromPrevious) < 180) {
                        if (referenceAngleFromPrevious < actualAngleFromPrevious) {
                            force1 = force1 * (-1);
                            force2 = force2 * (-1);
                        }
                        atomicAdd(&connectedCell->shared1.x, force1.x);
                        atomicAdd(&connectedCell->shared1.y, force1.y);
                        atomicAdd(&cell->connections[lastIndex].cell->shared1.x, force2.x);
                        atomicAdd(&cell->connections[lastIndex].cell->shared1.y, force2.y);
                        force = force - (force1 + force2);
                    }
                }
            }

            prevDisplacement = displacement;
        }
        atomicAdd(&cell->shared1.x, force.x);
        atomicAdd(&cell->shared1.y, force.y);
    }
}

__inline__ __device__ void CellProcessor::checkConnections(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        bool scheduleForDestruction = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectingCell = cell->connections[i].cell;

            auto displacement = connectingCell->absPos - cell->absPos;
            data.cellMap.correctDirection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.cellMaxBindingDistance) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
        }
    }
}

__inline__ __device__ void CellProcessor::verletPositionUpdate(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition =
        calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        cell->absPos = cell->absPos + cell->vel * cudaSimulationParameters.timestepSize
            + cell->shared1 * cudaSimulationParameters.timestepSize * cudaSimulationParameters.timestepSize / 2;
        data.cellMap.correctPosition(cell->absPos);
        cell->shared2 = cell->shared1;  //forces
        cell->shared1 = {0, 0};
    }
}

__inline__ __device__ void CellProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            cell->vel = {0, 0};
        } else {
            auto acceleration = (cell->shared1 + cell->shared2) / 2;
            cell->vel = cell->vel + acceleration * cudaSimulationParameters.timestepSize;
        }
    }
}

__inline__ __device__ void CellProcessor::livingStateTransition(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        auto livingState = atomicCAS(&cell->livingState, LivingState_JustReady, LivingState_Ready);
        if (livingState == LivingState_JustReady) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                atomicCAS(&connectedCell->livingState, LivingState_UnderConstruction, LivingState_JustReady);
            }
        }
        if (livingState == LivingState_Dying) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                if (connectedCell->cellFunction != CellFunction_None) {
                    atomicExch(&connectedCell->livingState, LivingState_Dying);
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyInnerFriction(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition =
        calcAllThreadsPartition(cells.getNumEntries());

    auto const innerFriction = cudaSimulationParameters.innerFriction;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        for (int index = 0; index < cell->numConnections; ++index) {
            auto connectingCell = cell->connections[index].cell;

            SystemDoubleLock lock;
            lock.init(&cell->locked, &connectingCell->locked);
            if (lock.tryLock()) {
                auto averageVel = (cell->vel + connectingCell->vel) / 2;
                cell->vel = cell->vel * (1.0f - innerFriction) + averageVel * innerFriction;
                connectingCell->vel = connectingCell->vel * (1.0f - innerFriction) + averageVel * innerFriction;
                lock.releaseLock();
            }
        }
    }
}

__inline__ __device__ void CellProcessor::applyFriction(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition =
        calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        auto friction = SpotCalculator::calcParameter(
            &SimulationParametersSpotValues::friction, &SimulationParametersSpotActivatedValues::friction, data, cell->absPos);
        cell->vel = cell->vel * (1.0f - friction);
    }
}

__inline__ __device__ void CellProcessor::radiation(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;

    auto partition =
        calcAllThreadsPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        if (data.numberGen1.random() < cudaSimulationParameters.radiationProb
            && (cell->energy > cudaSimulationParameters.highRadiationMinCellEnergy || cell->age > cudaSimulationParameters.radiationMinCellAge)) {
            auto radiationFactor = [&] {
                if (cell->energy < cudaSimulationParameters.highRadiationMinCellEnergy) {
                    return SpotCalculator::calcParameter(
                        &SimulationParametersSpotValues::radiationFactor, &SimulationParametersSpotActivatedValues::radiationFactor, data, cell->absPos);
                } else {
                    return cudaSimulationParameters.highRadiationFactor;
                }
            }();
            if (radiationFactor > 0) {

                auto pos = cell->absPos;
                if (cudaSimulationParameters.numParticleSources > 0) {
                    auto sourceIndex = data.numberGen1.random(cudaSimulationParameters.numParticleSources - 1);
                    pos.x = cudaSimulationParameters.particleSources[sourceIndex].posX;
                    pos.y = cudaSimulationParameters.particleSources[sourceIndex].posY;
                }
                float2 particleVel = cell->vel * cudaSimulationParameters.radiationVelocityMultiplier
                    + Math::unitVectorOfAngle(data.numberGen1.random() * 360) * cudaSimulationParameters.radiationVelocityPerturbation;
                float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
                data.cellMap.correctPosition(particlePos);

                auto const& cellEnergy = cell->energy;
                particlePos = particlePos - particleVel;  //because particle will still be moved in current time step
                auto radiationEnergy = cellEnergy * radiationFactor;
                radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProb;
                radiationEnergy = 2 * radiationEnergy * data.numberGen1.random();
                if (cellEnergy > 1) {
                    if (radiationEnergy > cellEnergy - 1) {
                        radiationEnergy = cellEnergy - 1;
                    }
                    cell->energy -= radiationEnergy;

                    ObjectFactory factory;
                    factory.init(&data);
                    factory.createParticle(radiationEnergy, particlePos, particleVel, cell->color);
                }
            }
        }
    }
}

__inline__ __device__ void CellProcessor::decay(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition =
        calcAllThreadsPartition(cells.getNumEntries());

    for (auto index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }

        auto cellMinEnergy = SpotCalculator::calcParameter(
            &SimulationParametersSpotValues::cellMinEnergy, &SimulationParametersSpotActivatedValues::cellMinEnergy, data, cell->absPos);
        auto cellMaxBindingEnergy = SpotCalculator::calcParameter(
            &SimulationParametersSpotValues::cellMaxBindingEnergy, &SimulationParametersSpotActivatedValues::cellMaxBindingEnergy, data, cell->absPos);

        bool decay = false;
        if (cell->livingState == LivingState_Dying) {
            if (data.numberGen1.random() < cudaSimulationParameters.clusterDecayProb) {
                CellConnectionProcessor::scheduleDeleteCell(data, index);
                decay = true;
            }
        }
        if (cell->energy < cellMinEnergy) {
            if (cudaSimulationParameters.clusterDecay) {
                cell->livingState = LivingState_Dying;
            } else {
                CellConnectionProcessor::scheduleDeleteCell(data, index);
                decay = true;
            }
        } else if (cell->energy > cellMaxBindingEnergy) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            decay = true;
        }

        if (decay && cell->livingState != LivingState_UnderConstruction) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& connectedCell = cell->connections[i].cell;
                if (connectedCell->livingState == LivingState_UnderConstruction) {
                    connectedCell->livingState = LivingState_JustReady;
                }
            }
        }
    }
}

