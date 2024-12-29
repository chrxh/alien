#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "EngineInterface/CellFunctionConstants.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "ObjectFactory.cuh"
#include "Map.cuh"
#include "Physics.cuh"
#include "CellConnectionProcessor.cuh"
#include "GenomeDecoder.cuh"
#include "RadiationProcessor.cuh"
#include "SpotCalculator.cuh"

class CellProcessor
{
public:
    __inline__ __device__ static void init(SimulationData& data);
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void clearDensityMap(SimulationData& data);
    __inline__ __device__ static void fillDensityMap(SimulationData& data);

    __inline__ __device__ static void calcFluidForces_reconnectCells_correctOverlap(SimulationData& data);
    __inline__ __device__ static void calcCollisions_reconnectCells_correctOverlap(SimulationData& data);
    __inline__ __device__ static void checkForces(SimulationData& data);
    __inline__ __device__ static void applyForces(SimulationData& data);  //prerequisite: data from calcCollisions_reconnectCells_correctOverlap

    __inline__ __device__ static void calcConnectionForces(SimulationData& data, bool considerAngles);
    __inline__ __device__ static void checkConnections(SimulationData& data);
    __inline__ __device__ static void verletPositionUpdate(SimulationData& data);
    __inline__ __device__ static void verletVelocityUpdate(SimulationData& data);

    __inline__ __device__ static void aging(SimulationData& data);
    __inline__ __device__ static void livingStateTransition_calcFutureState(SimulationData& data);
    __inline__ __device__ static void livingStateTransition_applyNextState(SimulationData& data);

    __inline__ __device__ static void applyInnerFriction(SimulationData& data);
    __inline__ __device__ static void applyFriction(SimulationData& data);

    __inline__ __device__ static void radiation(SimulationData& data);
    __inline__ __device__ static void decay(SimulationData& data);

    __inline__ __device__ static void resetDensity(SimulationData& data);

private:
    static auto constexpr MaxBarrierCellsForCollision = 10;
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

        cell->shared1 = {0, 0};
        cell->nextCell = nullptr;
        cell->tag = 0;
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
    data.preprocessedSimulationData.densityMap.clear();
}

__inline__ __device__ void CellProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        data.preprocessedSimulationData.densityMap.addCell(data.timestep, data.objects.cellPointers.at(index));
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

__inline__ __device__ void CellProcessor::calcFluidForces_reconnectCells_correctOverlap(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto blockPartition = calcBlockPartition(cells.getNumEntries());
    auto const& smoothingLength = cudaSimulationParameters.motionData.fluidMotion.smoothingLength;

    for (int cellIndex = blockPartition.startIndex; cellIndex <= blockPartition.endIndex; ++cellIndex) {
        auto& cell = cells.at(cellIndex);

        __shared__ float2 F_pressure;
        __shared__ float2 F_viscosity;
        __shared__ float2 cellPosDelta;
        __shared__ float density;
        __shared__ float cellMaxBindingEnergy;
        __shared__ float cellFusionVelocity;

        __shared__ int scanLength;
        __shared__ int2 cellPosInt;

        __shared__ Cell* barrierCells[MaxBarrierCellsForCollision];
        __shared__ int numBarrierCells;

        if (threadIdx.x == 0) {
            F_pressure = {0, 0};
            F_viscosity = {0, 0};
            cellPosDelta = {0, 0};
            density = 0;
            cellMaxBindingEnergy = SpotCalculator::calcParameter(
                &SimulationParametersZoneValues::cellMaxBindingEnergy, &SimulationParametersZoneActivatedValues::cellMaxBindingEnergy, data, cell->pos);
            cellFusionVelocity = SpotCalculator::calcParameter(
                &SimulationParametersZoneValues::cellFusionVelocity, &SimulationParametersZoneActivatedValues::cellFusionVelocity, data, cell->pos);

            int radiusInt = ceilf(smoothingLength * 2);
            scanLength = radiusInt * 2 + 1;
            cellPosInt = {floorInt(cell->pos.x) - radiusInt, floorInt(cell->pos.y) - radiusInt};

            numBarrierCells = 0;
        }
        __syncthreads();

        int2 scanPos{cellPosInt.x + (toInt(threadIdx.x) % scanLength), cellPosInt.y + (toInt(threadIdx.x) / scanLength)};
        data.cellMap.correctPosition(scanPos);
        auto otherCell = data.cellMap.getFirst(scanPos);
        for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
            if (!otherCell) {
                break;
            }
            auto posDelta = cell->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);
            auto distance = Math::length(posDelta);

            if (otherCell->barrier && distance <= smoothingLength * 2 && cell->detached + otherCell->detached != 1) {
                auto index = atomicAdd(&numBarrierCells, 1);
                if (index < MaxBarrierCellsForCollision) {
                    barrierCells[index] = otherCell;
                }
            }

            if (!otherCell->barrier && distance <= smoothingLength * 2 && cell->detached + otherCell->detached != 1) {

                //calc density
                atomicAdd_block(&density, calcKernel(distance / smoothingLength) / (smoothingLength * smoothingLength));

                if (cell != otherCell) {

                    //overlap correction
                    if (!cell->barrier && distance < cudaSimulationParameters.cellMinDistance) {
                        atomicAdd_block(&cellPosDelta.x, posDelta.x * cudaSimulationParameters.cellMinDistance / 5);
                        atomicAdd_block(&cellPosDelta.y, posDelta.y * cudaSimulationParameters.cellMinDistance / 5);
                    }

                    bool isConnected = false;
                    for (int i = 0; i < cell->numConnections; ++i) {
                        auto const& connectedCell = cell->connections[i].cell;
                        if (connectedCell == otherCell) {
                            isConnected = true;
                        }
                    }
                    if (!isConnected) {

                        //calc forces: for simplicity pressure = density
                        auto velDelta = cell->vel - otherCell->vel;
                        auto const& cellPressure = cell->density;            //optimization: using the density from last time step
                        auto const& otherCellPressure = otherCell->density;  //optimization: using the density from last time step
                        auto factor = (cellPressure / (cell->density * cell->density) + otherCellPressure / (otherCell->density * otherCell->density));

                        if (abs(distance) > NEAR_ZERO) {
                            float kernel_d = calcKernel_d(distance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                            auto F_pressureDelta = posDelta / (-distance) * factor * kernel_d;
                            atomicAdd_block(&F_pressure.x, F_pressureDelta.x);
                            atomicAdd_block(&F_pressure.y, F_pressureDelta.y);

                            auto F_viscosityDelta = velDelta / otherCell->density * distance * kernel_d / (distance * distance + 0.25f);
                            atomicAdd_block(&F_viscosity.x, F_viscosityDelta.x);
                            atomicAdd_block(&F_viscosity.y, F_viscosityDelta.y);
                        }

                        //fusion
                        if (Math::length(velDelta) >= cellFusionVelocity && cell->numConnections < cell->maxConnections
                            && otherCell->numConnections < otherCell->maxConnections && cell->energy <= cellMaxBindingEnergy
                            && otherCell->energy <= cellMaxBindingEnergy && !cell->barrier && !otherCell->barrier) {
                            CellConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                        }
                    }
                }
            }
            otherCell = otherCell->nextCell;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            //calculate barrier forces
            numBarrierCells = min(MaxBarrierCellsForCollision, numBarrierCells);
            if (numBarrierCells > 0) {
                Cell* closestBarrierCell = nullptr;
                float closestBarrierCellDistance;
                for (int i = 0; i < numBarrierCells; ++i) {
                    auto const& barrierCell = barrierCells[i];
                    auto distance = data.cellMap.getDistance(cell->pos, barrierCell->pos);
                    if (!closestBarrierCell || distance < closestBarrierCellDistance) {
                        closestBarrierCell = barrierCell;
                        closestBarrierCellDistance = distance;
                    }
                }

                float2 r{0,0};
                if (closestBarrierCell->numConnections <= 1) {
                    r = data.cellMap.getCorrectedDirection(cell->pos - closestBarrierCell->pos);
                } else {
                    auto angleToCell = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell->pos - closestBarrierCell->pos));
                    auto numConnections = closestBarrierCell->numConnections;
                    for (int i = 0; i < numConnections; ++i) {
                        auto otherCell1 = closestBarrierCell->connections[i].cell;
                        auto otherCell2 = closestBarrierCell->connections[(i + 1) % numConnections].cell;
                        auto angleToOtherCell1 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell1->pos - closestBarrierCell->pos));
                        auto angleToOtherCell2 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell2->pos - closestBarrierCell->pos));
                        if (Math::isAngleInBetween(angleToOtherCell1, angleToOtherCell2, angleToCell)) {
                            r = otherCell2->pos - otherCell1->pos;
                            Math::rotateQuarterCounterClockwise(r);
                            break;
                        }
                    }
                }
                auto vr = cell->vel - closestBarrierCell->vel;
                auto dot_vr_r = Math::dot(vr, r);

                if (dot_vr_r < 0) {
                    auto truncated_r_squared = max(0.05f, Math::lengthSquared(r));
                    auto truncated_distance = max(0.05f, closestBarrierCellDistance);
                    cell->shared1 += (vr - r * 2 * dot_vr_r / truncated_r_squared + closestBarrierCell->vel - cell->vel) / truncated_distance;
                }
            }

            cell->pos += cellPosDelta;
            cell->shared1 += F_pressure * cudaSimulationParameters.motionData.fluidMotion.pressureStrength
                + F_viscosity * cudaSimulationParameters.motionData.fluidMotion.viscosityStrength;
            cell->shared2.x = density;
        }
        __syncthreads();
    }
}

__inline__ __device__ void CellProcessor::calcCollisions_reconnectCells_correctOverlap(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        data.cellMap.executeForEach(
            cell->pos, cudaSimulationParameters.motionData.collisionMotion.cellMaxCollisionDistance, cell->detached, [&](auto const& otherCell) {
                if (otherCell == cell) {
                    return;
                }

                auto posDelta = cell->pos - otherCell->pos;
                data.cellMap.correctDirection(posDelta);

                auto distance = Math::length(posDelta);

                //overlap correction
                if (!cell->barrier) {
                    if (distance < cudaSimulationParameters.cellMinDistance) {
                        cell->pos += posDelta * cudaSimulationParameters.cellMinDistance / 5;
                    }
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

                    //collision algorithm
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

                    //fusion
                    auto cellMaxBindingEnergy = SpotCalculator::calcParameter(
                        &SimulationParametersZoneValues::cellMaxBindingEnergy,
                        &SimulationParametersZoneActivatedValues::cellMaxBindingEnergy,
                        data,
                        cell->pos);

                    auto cellFusionVelocity = SpotCalculator::calcParameter(
                        &SimulationParametersZoneValues::cellFusionVelocity,
                        &SimulationParametersZoneActivatedValues::cellFusionVelocity,
                        data,
                        cell->pos);

                    if (cell->numConnections < cell->maxConnections && otherCell->numConnections < otherCell->maxConnections
                        && Math::length(velDelta) >= cellFusionVelocity
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
        cell->density = cell->shared2.x;
        if (cell->barrier) {
            continue;
        }

        if (Math::length(cell->shared1) > SpotCalculator::calcParameter(
                &SimulationParametersZoneValues::cellMaxForce, &SimulationParametersZoneActivatedValues::cellMaxForce, data, cell->pos, cell->color)) {
            if (data.numberGen1.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
                CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
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

        cell->vel += cell->shared1;
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
        float2 prevDisplacement = cell->connections[cell->numConnections - 1].cell->pos - cell->pos;
        data.cellMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = cell->stiffness * cell->stiffness;

        auto numConnections = cell->numConnections;
        for (int i = 0; i < numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            auto connectedCellStiffnessSquared = connectedCell->stiffness * connectedCell->stiffness;

            auto displacement = connectedCell->pos - cell->pos;
            data.cellMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = cell->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::normalized(displacement) * deviation * (cellStiffnessSquared + connectedCellStiffnessSquared) / 6;

            if (considerAngles && (numConnections > 2 || (numConnections == 2 && i == 0))) {

                auto lastIndex = (i + numConnections - 1) % numConnections;
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
                    if (actualAngleFromPrevious < 0) {
                        continue;
                    }
                    auto referenceAngleFromPrevious = cell->connections[i].angleFromPrevious;

                    auto strength = abs(referenceAngleFromPrevious - actualAngleFromPrevious) / 2000 * cellStiffnessSquared;

                    auto force1 = Math::normalized(displacement) / max(Math::length(displacement), cudaSimulationParameters.cellMinDistance) * strength;
                    Math::rotateQuarterClockwise(force1);

                    auto force2 =
                        Math::normalized(prevDisplacement) / max(Math::length(prevDisplacement), cudaSimulationParameters.cellMinDistance) * strength;
                    Math::rotateQuarterCounterClockwise(force2);

                    if (referenceAngleFromPrevious < actualAngleFromPrevious) {
                        force1 = force1 * (-1);
                        force2 = force2 * (-1);
                    }
                    if (!connectedCell->barrier) {
                        atomicAdd(&connectedCell->shared1.x, force1.x);
                        atomicAdd(&connectedCell->shared1.y, force1.y);
                    }
                    auto otherConnectedCell = cell->connections[lastIndex].cell;
                    if (!otherConnectedCell->barrier) {
                        atomicAdd(&otherConnectedCell->shared1.x, force2.x);
                        atomicAdd(&otherConnectedCell->shared1.y, force2.y);
                    }
                    force -= force1 + force2;
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
            auto connectedCell = cell->connections[i].cell;

            auto displacement = connectedCell->pos - cell->pos;
            data.cellMap.correctDirection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.cellMaxBindingDistance[cell->color]) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            for (int i = 0; i < cell->numConnections; ++i) {
                auto connectedCell = cell->connections[i].cell;
                connectedCell->livingState = LivingState_Detaching;
            }
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
            cell->pos += cell->vel * cudaSimulationParameters.timestepSize;
            data.cellMap.correctPosition(cell->pos);
        } else {
            cell->pos += cell->vel * cudaSimulationParameters.timestepSize
                + cell->shared1 * cudaSimulationParameters.timestepSize * cudaSimulationParameters.timestepSize / 2;
            data.cellMap.correctPosition(cell->pos);
            cell->shared2 = cell->shared1;  //forces
            cell->shared1 = {0, 0};
        }
    }
}

__inline__ __device__ void CellProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto const partition = calcAllThreadsPartition(cells.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->barrier) {
            continue;
        }
        auto acceleration = (cell->shared1 + cell->shared2) / 2;
        cell->vel += acceleration * cudaSimulationParameters.timestepSize;
    }
}

__inline__ __device__ void CellProcessor::aging(SimulationData& data)
{
    auto const partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (cell->barrier) {
            continue;
        }
        ++cell->age;


        if (cudaSimulationParameters.features.cellColorTransitionRules) {
            int transitionDuration;
            int targetColor;
            auto color = calcMod(cell->color, MAX_COLORS);
            auto spotIndex = SpotCalculator::getFirstMatchingSpotOrBase(data, cell->pos, &SimulationParametersZoneActivatedValues::cellColorTransition);
            if (spotIndex == -1) {
                transitionDuration = cudaSimulationParameters.baseValues.cellColorTransitionDuration[color];
                targetColor = cudaSimulationParameters.baseValues.cellColorTransitionTargetColor[color];
            } else {
                transitionDuration = cudaSimulationParameters.zone[spotIndex].values.cellColorTransitionDuration[color];
                targetColor = cudaSimulationParameters.zone[spotIndex].values.cellColorTransitionTargetColor[color];
            }
            if (transitionDuration > 0 && cell->age > transitionDuration) {
                cell->color = targetColor;
                cell->age = 0;
            }
        }
        if (cell->livingState == LivingState_Ready && cell->activationTime > 0) {
            --cell->activationTime;
        }
    }
}


__inline__ __device__ void CellProcessor::livingStateTransition_calcFutureState(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        bool isSameCreatureNeighborDetaching = false;
        bool isOtherCreatureNeighborDetaching = false;
        bool isSameCreatureNeighborReviving= false;
        bool isNeighborActivating = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto const& connectedCell = cell->connections[i].cell;
            if (connectedCell->creatureId == cell->creatureId) {
                auto connectedLivingState = connectedCell->livingState;
                if (connectedLivingState == LivingState_Detaching) {
                    isSameCreatureNeighborDetaching = true;
                } else if (connectedLivingState == LivingState_Reviving) {
                    isSameCreatureNeighborReviving = true;
                } else if (connectedLivingState == LivingState_Activating) {
                    isNeighborActivating = true;
                }
            } else {
                if (connectedCell->livingState == LivingState_Detaching) {
                    isOtherCreatureNeighborDetaching = true;
                }
            }
        }

        auto origLivingState = cell->livingState;
        auto livingState = origLivingState;

        if (cell->barrier) {
            livingState = LivingState_Ready;
        } else if (origLivingState == LivingState_Activating) {
            livingState = LivingState_Ready;
            if (cudaSimulationParameters.features.cellAgeLimiter && cudaSimulationParameters.cellResetAgeAfterActivation) {
                atomicExch(&cell->age, 0);
            }
        } else if (origLivingState == LivingState_Reviving) {
            livingState = LivingState_Ready;
        } else if (origLivingState == LivingState_UnderConstruction) {
            if (isNeighborActivating) {
                livingState = LivingState_Activating;
            }
            if (isOtherCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences != CellDeathConsquences_None) {
                livingState = LivingState_Detaching;
            }
        } else if (origLivingState == LivingState_Detaching) {
            if (isSameCreatureNeighborReviving && cudaSimulationParameters.cellDeathConsequences == CellDeathConsquences_DetachedPartsDie) {
                livingState = LivingState_Reviving;
            }
            if (cudaSimulationParameters.cellDeathConsequences == CellDeathConsquences_None) {
                livingState = LivingState_Ready;
            }
        } else if (origLivingState == LivingState_Ready) {
            if (isSameCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences != CellDeathConsquences_None) {
                if (cudaSimulationParameters.cellDeathConsequences == CellDeathConsquences_DetachedPartsDie && cell->cellFunction == CellFunction_Constructor
                    && GenomeDecoder::containsSelfReplication(cell->cellFunctionData.constructor)) {
                    livingState = LivingState_Reviving;
                } else {
                    livingState = LivingState_Detaching;
                }
            }
        }
        cell->tag = livingState;
    }
}

__inline__ __device__ void CellProcessor::livingStateTransition_applyNextState(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        cell->livingState = cell->tag;
        cell->tag = 0;
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
            if (connectingCell->barrier) {
                continue;
            }
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
            &SimulationParametersZoneValues::friction, &SimulationParametersZoneActivatedValues::friction, data, cell->pos);
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
        if (data.numberGen1.random() < cudaSimulationParameters.radiationProb) {

            auto radiationFactor = 0.0f;
            if (cell->energy > cudaSimulationParameters.highRadiationMinCellEnergy[cell->color]) {
                radiationFactor += cudaSimulationParameters.highRadiationFactor[cell->color];
            }
            if (cell->age > cudaSimulationParameters.radiationMinCellAge[cell->color]) {
                radiationFactor += SpotCalculator::calcParameter(
                    &SimulationParametersZoneValues::radiationCellAgeStrength,
                    &SimulationParametersZoneActivatedValues::radiationCellAgeStrength,
                    data,
                    cell->pos,
                    cell->color);
            }

            if (radiationFactor > 0) {

                auto const& cellEnergy = cell->energy;
                auto energyLoss = cellEnergy * radiationFactor;
                energyLoss = energyLoss / cudaSimulationParameters.radiationProb;
                energyLoss = 2 * energyLoss * data.numberGen1.random();
                if (cellEnergy > 1) {

                    float2 particleVel = cell->vel * cudaSimulationParameters.radiationVelocityMultiplier
                        + Math::unitVectorOfAngle(data.numberGen1.random() * 360) * cudaSimulationParameters.radiationVelocityPerturbation;
                    float2 particlePos = cell->pos + Math::normalized(particleVel) * 1.5f
                        - particleVel;  //"- particleVel" because particle will still be moved in current time step
                    data.cellMap.correctPosition(particlePos);
                    if (energyLoss > cellEnergy - 1) {
                        energyLoss = cellEnergy - 1;
                    }
                    RadiationProcessor::createEnergyParticle(data, particlePos, particleVel, cell->color, energyLoss);
                    cell->energy -= energyLoss;
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
        auto cellMaxBindingEnergy = SpotCalculator::calcParameter(
            &SimulationParametersZoneValues::cellMaxBindingEnergy, &SimulationParametersZoneActivatedValues::cellMaxBindingEnergy, data, cell->pos);
        if (cell->energy > cellMaxBindingEnergy) {
            CellConnectionProcessor::scheduleDeleteAllConnections(data, cell);
        }

        auto cellMinEnergy = SpotCalculator::calcParameter(
            &SimulationParametersZoneValues::cellMinEnergy, &SimulationParametersZoneActivatedValues::cellMinEnergy, data, cell->pos, cell->color);

        if (cell->livingState == LivingState_Dying || cell->livingState == LivingState_Detaching) {
            auto cellDeathProbability = SpotCalculator::calcParameter(
                &SimulationParametersZoneValues::cellDeathProbability,
                &SimulationParametersZoneActivatedValues::cellDeathProbability,
                data,
                cell->pos,
                cell->color);
            if (data.numberGen1.random() < cellDeathProbability) {
                CellConnectionProcessor::scheduleDeleteCell(data, index);
            }
        }

        bool cellDestruction = false;
        if (cell->energy < cellMinEnergy) {
            cellDestruction = true;
        }

        auto cellMaxAge = cudaSimulationParameters.cellMaxAge[cell->color];
        if (cudaSimulationParameters.features.cellAgeLimiter && cudaSimulationParameters.cellInactiveMaxAgeActivated && cell->mutationId != 1
            && cell->cellFunctionUsed == CellFunctionUsed_No && cell->livingState == LivingState_Ready && cell->activationTime == 0) {
            bool adjacentCellsUsed = false;
            for (int i = 0; i < cell->numConnections; ++i) {
                if (cell->connections[i].cell->cellFunctionUsed == CellFunctionUsed_Yes) {
                    adjacentCellsUsed = true;
                    break;
                }
            }
            if (!adjacentCellsUsed) {
                auto cellInactiveMaxAge = SpotCalculator::calcParameter(
                    &SimulationParametersZoneValues::cellInactiveMaxAge,
                    &SimulationParametersZoneActivatedValues::cellInactiveMaxAge,
                    data,
                    cell->pos,
                    cell->color);

                cellMaxAge = toInt(cellInactiveMaxAge);
            }
        }
        if (cudaSimulationParameters.features.cellAgeLimiter && cudaSimulationParameters.cellEmergentMaxAgeActivated && cell->mutationId == 1) {
            cellMaxAge = cudaSimulationParameters.cellEmergentMaxAge[cell->color];
        }
        if (cellMaxAge > 0 && cell->age > cellMaxAge) {
            cellDestruction = true;
        }

        if (cellDestruction) {
            auto orig = atomicExch(&cell->livingState, LivingState_Dying);
            if (orig != LivingState_Dying) {
                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const& connectedCell = cell->connections[i].cell;
                    auto origConnected = atomicExch(&connectedCell->livingState, LivingState_Detaching);
                    if (origConnected == LivingState_Dying) {
                        atomicExch(&connectedCell->livingState, LivingState_Dying);
                    }
                }
            }
            //else {
            //    if (data.numberGen1.random() < cudaSimulationParameters.clusterDecayProb[cell->color]) {
            //        CellConnectionProcessor::scheduleDeleteCell(data, index);
            //    }
            //}
        }
    }
}

__inline__ __device__ void CellProcessor::resetDensity(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->density = 1.0f;
    }
}
