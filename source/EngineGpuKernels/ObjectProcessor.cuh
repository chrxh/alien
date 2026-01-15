#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <EngineInterface/CellTypeConstants.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ConstructorHelper.cuh"
#include "Map.cuh"
#include "MuscleProcessor.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "Physics.cuh"
#include "EnergyParticleProcessor.cuh"
#include "TO.cuh"

namespace cg = cooperative_groups;

class ObjectProcessor
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

    __inline__ __device__ static void calcConnectionForces(SimulationData& data, bool calcAngularForces);
    __inline__ __device__ static void checkConnections(SimulationData& data);
    __inline__ __device__ static void verletPositionUpdate(SimulationData& data);
    __inline__ __device__ static void verletVelocityUpdate(SimulationData& data);

    __inline__ __device__ static void aging(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_calcFutureState(SimulationData& data);
    __inline__ __device__ static void cellStateTransition_applyNextState(SimulationData& data);
    __inline__ __device__ static void frontAngleUpdate_calcFutureValue(SimulationData& data);
    __inline__ __device__ static void frontAngleUpdate_applyFutureValue(SimulationData& data);

    __inline__ __device__ static void applyInnerFriction(SimulationData& data);
    __inline__ __device__ static void applyFriction(SimulationData& data);

    __inline__ __device__ static void radiation(SimulationData& data);
    __inline__ __device__ static void decay(SimulationData& data);

    __inline__ __device__ static void resetDensity(SimulationData& data);

    __inline__ __device__ static void updateCellEvents(SimulationData& data);

    __inline__ __device__ static void performEnergyFlow(SimulationData& data);

private:
    static auto constexpr MaxBarrierCellsForCollision = 10;
    static auto constexpr FrontAngleId_NoUpdate = VALUE_NOT_SET_FLOAT;

    __inline__ __device__ static float getInitialAngelSpan(Object* cell, int connectionIndex1, int connectionIndex2);
    __inline__ __device__ static float getInitialAngelSpan(Object* cell, Object* connectedCell1, Object* connectedCell2);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ObjectProcessor::init(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);

        object->shared1 = {0, 0};
        object->nextCell = nullptr;
        object->tempValue.as_uint64 = 0;
    }
}

__inline__ __device__ void ObjectProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcBlockPartition(data.entities.objects.getNumEntries());
    Object** cellPointers = &data.entities.objects.at(partition.startIndex);
    data.cellMap.set_block(partition.numElements(), cellPointers);
}

__inline__ __device__ void ObjectProcessor::clearDensityMap(SimulationData& data)
{
    data.preprocessedSimulationData.densityMap.clear();
}

__inline__ __device__ void ObjectProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto cell = data.entities.objects.at(index);
        if (object->cellType == CellType_Free) {
            data.preprocessedSimulationData.densityMap.addFreeCell(cell);
        } else if (object->cellType == CellType_Structure) {
            data.preprocessedSimulationData.densityMap.addStructureCell(cell);
        }
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

__inline__ __device__ void ObjectProcessor::calcFluidForces_reconnectCells_correctOverlap(SimulationData& data)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto& cells = data.entities.objects;
    auto blockPartition = calcBlockPartition(cells.getNumEntries());
    auto const& smoothingLength = cudaSimulationParameters.smoothingLength.value;

    for (int cellIndex = blockPartition.startIndex; cellIndex <= blockPartition.endIndex; ++cellIndex) {
        auto& object = cells.at(cellIndex);

        __shared__ float cellMaxBindingEnergy;
        __shared__ float cellFusionVelocity;

        __shared__ int scanLength;
        __shared__ int2 cellPosInt;

        __shared__ Object* fixedCells[MaxBarrierCellsForCollision];
        __shared__ int numFixedCells;

        __shared__ float2 F_pressure;
        __shared__ float2 F_viscosity;
        __shared__ float2 cellPosDelta;
        __shared__ float density;

        if (block.thread_rank() == 0) {
            cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, object->pos);
            cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.cellFusionVelocity, data, object->pos);

            int radiusInt = ceilf(smoothingLength * 2);
            scanLength = radiusInt * 2 + 1;
            cellPosInt = {floorInt(object->pos.x) - radiusInt, floorInt(object->pos.y) - radiusInt};

            numFixedCells = 0;
            F_pressure = {0, 0};
            F_viscosity = {0, 0};
            cellPosDelta = {0, 0};
            density = 0;
        }
        block.sync();

        // Per-thread accumulators
        float2 localF_pressure = {0, 0};
        float2 localF_viscosity = {0, 0};
        float2 localCellPosDelta = {0, 0};
        float localDensity = 0;

        int2 scanPos{cellPosInt.x + (toInt(block.thread_rank()) % scanLength), cellPosInt.y + (toInt(block.thread_rank()) / scanLength)};
        data.cellMap.correctPosition(scanPos);
        auto otherCell = data.cellMap.getFirst(scanPos);
        for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
            if (!otherCell) {
                break;
            }
            auto posDelta = object->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);
            auto adaptedDistance = Math::length(posDelta);
            auto origDistance = adaptedDistance;
            if (object->isSameCreature(otherCell) && (object->numConnections < 3 || otherCell->numConnections < 3)) {
                adaptedDistance *= 2.0f;  // Reduce range of cell repulsion within creature by scaling distance
            }

            if (otherCell->fixed && adaptedDistance <= smoothingLength * 2 && object->detached + otherCell->detached != 1) {
                auto index = atomicAdd(&numFixedCells, 1);
                if (index < MaxBarrierCellsForCollision) {
                    fixedCells[index] = otherCell;
                }
            }

            if (!otherCell->fixed && adaptedDistance <= smoothingLength * 2 && object->detached + otherCell->detached != 1) {

                // Calc density
                localDensity += calcKernel(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength);

                if (cell != otherCell) {

                    // Overlap correction
                    if (!object->fixed && origDistance < cudaSimulationParameters.minCellDistance.value) {
                        localCellPosDelta.x += posDelta.x * cudaSimulationParameters.minCellDistance.value / 5;
                        localCellPosDelta.y += posDelta.y * cudaSimulationParameters.minCellDistance.value / 5;
                    }

                    auto velDelta = object->vel - otherCell->vel;
                    bool isConnected = false;
                    for (int i = 0; i < object->numConnections; ++i) {
                        auto const& connectedCell = object->connections[i].cell;
                        if (connectedCell == otherCell) {
                            isConnected = true;
                        }
                    }
                    if (!isConnected) {

                        // Calc forces: for simplicity pressure = density
                        auto const& cellPressure = object->density;            // Optimization: using the density from last time step
                        auto const& otherCellPressure = otherCell->density;  // Optimization: using the density from last time step
                        auto factor = cellPressure / (object->density * object->density) + otherCellPressure / (otherCell->density * otherCell->density);

                        if (adaptedDistance > NEAR_ZERO) {
                            float kernel_d = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                            auto F_pressureDelta = posDelta / (-adaptedDistance) * factor * kernel_d;
                            localF_pressure.x += F_pressureDelta.x;
                            localF_pressure.y += F_pressureDelta.y;

                            auto F_viscosityDelta = velDelta / otherCell->density * adaptedDistance * kernel_d / (adaptedDistance * adaptedDistance + 0.25f);
                            localF_viscosity.x += F_viscosityDelta.x;
                            localF_viscosity.y += F_viscosityDelta.y;
                        }
                    }

                    // Fusion
                    if (Math::length(velDelta) >= cellFusionVelocity && object->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS
                        && (object->sticky || otherCell->sticky) && object->usableEnergy <= cellMaxBindingEnergy && otherCell->usableEnergy <= cellMaxBindingEnergy
                        && !object->fixed && !otherCell->fixed) {
                        ObjectConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                    }
                }
            }
            otherCell = otherCell->nextCell;
        }

        // Warp-level reduction followed by atomic accumulation across warps
        float sumF_pressure_x = cg::reduce(warp, localF_pressure.x, cg::plus<float>());
        float sumF_pressure_y = cg::reduce(warp, localF_pressure.y, cg::plus<float>());
        float sumF_viscosity_x = cg::reduce(warp, localF_viscosity.x, cg::plus<float>());
        float sumF_viscosity_y = cg::reduce(warp, localF_viscosity.y, cg::plus<float>());
        float sumCellPosDelta_x = cg::reduce(warp, localCellPosDelta.x, cg::plus<float>());
        float sumCellPosDelta_y = cg::reduce(warp, localCellPosDelta.y, cg::plus<float>());
        float sumDensity = cg::reduce(warp, localDensity, cg::plus<float>());

        // Each warp leader adds its warp's sum to shared memory
        if (warp.thread_rank() == 0) {
            atomicAdd_block(&F_pressure.x, sumF_pressure_x);
            atomicAdd_block(&F_pressure.y, sumF_pressure_y);
            atomicAdd_block(&F_viscosity.x, sumF_viscosity_x);
            atomicAdd_block(&F_viscosity.y, sumF_viscosity_y);
            atomicAdd_block(&cellPosDelta.x, sumCellPosDelta_x);
            atomicAdd_block(&cellPosDelta.y, sumCellPosDelta_y);
            atomicAdd_block(&density, sumDensity);
        }
        block.sync();

        if (block.thread_rank() == 0) {
            // Calculate fixed forces
            numFixedCells = min(MaxBarrierCellsForCollision, numFixedCells);
            if (numFixedCells > 0) {
                Object* closestFixedCell = nullptr;
                float closestFixedCellDistance;
                for (int i = 0; i < numFixedCells; ++i) {
                    auto const& fixedCell = fixedCells[i];
                    auto distance = data.cellMap.getDistance(object->pos, fixedCell->pos);
                    if (!closestFixedCell || distance < closestFixedCellDistance) {
                        closestFixedCell = fixedCell;
                        closestFixedCellDistance = distance;
                    }
                }

                float2 r{0, 0};
                if (closestFixedCell->numConnections <= 1) {
                    r = data.cellMap.getCorrectedDirection(object->pos - closestFixedCell->pos);
                } else {
                    auto angleToCell = Math::angleOfVector(data.cellMap.getCorrectedDirection(object->pos - closestFixedCell->pos));
                    auto numConnections = closestFixedCell->numConnections;
                    for (int i = 0; i < numConnections; ++i) {
                        auto otherCell1 = closestFixedCell->connections[i].cell;
                        auto otherCell2 = closestFixedCell->connections[(i + 1) % numConnections].cell;
                        auto angleToOtherCell1 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell1->pos - closestFixedCell->pos));
                        auto angleToOtherCell2 = Math::angleOfVector(data.cellMap.getCorrectedDirection(otherCell2->pos - closestFixedCell->pos));
                        if (Math::isAngleInBetween(angleToOtherCell1, angleToOtherCell2, angleToCell)) {
                            r = otherCell2->pos - otherCell1->pos;
                            Math::rotateQuarterCounterClockwise(r);
                            break;
                        }
                    }
                }
                auto vr = object->vel - closestFixedCell->vel;
                auto dot_vr_r = Math::dot(vr, r);

                if (dot_vr_r < 0) {
                    auto truncated_r_squared = max(0.05f, Math::lengthSquared(r));
                    auto truncated_distance = max(0.05f, closestFixedCellDistance);
                    object->shared1 += (vr - r * 2 * dot_vr_r / truncated_r_squared + closestFixedCell->vel - object->vel) / truncated_distance;
                }
            }

            object->pos += cellPosDelta;
            object->shared1 += F_pressure * cudaSimulationParameters.pressureStrength.value + F_viscosity * cudaSimulationParameters.viscosityStrength.value;
            object->shared2.x = density;
        }
        block.sync();
    }
}

__inline__ __device__ void ObjectProcessor::calcCollisions_reconnectCells_correctOverlap(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        data.cellMap.executeForEach(object->pos, cudaSimulationParameters.maxCollisionDistance.value, object->detached, [&](auto const& otherCell) {
            if (otherCell == cell) {
                return;
            }

            auto posDelta = object->pos - otherCell->pos;
            data.cellMap.correctDirection(posDelta);

            auto distance = Math::length(posDelta);

            //overlap correction
            if (!object->fixed) {
                if (distance < cudaSimulationParameters.minCellDistance.value) {
                    object->pos += posDelta * cudaSimulationParameters.minCellDistance.value / 5;
                }
            }

            bool alreadyConnected = false;
            for (int i = 0; i < object->numConnections; ++i) {
                auto const& connectedCell = object->connections[i].cell;
                if (connectedCell == otherCell) {
                    alreadyConnected = true;
                    break;
                }
            }

            if (!alreadyConnected) {

                //collision algorithm
                auto velDelta = object->vel - otherCell->vel;
                auto isApproaching = Math::dot(posDelta, velDelta) < 0;
                auto fixedFactor = object->fixed ? 2 : 1;

                if (Math::length(object->vel) > 0.5f && isApproaching) {
                    auto distanceSquared = distance * distance + 0.25f;
                    auto force = posDelta * Math::dot(velDelta, posDelta) / (-2 * distanceSquared) * fixedFactor;
                    atomicAdd(&object->shared1.x, force.x);
                    atomicAdd(&object->shared1.y, force.y);
                    atomicAdd(&otherCell->shared1.x, -force.x);
                    atomicAdd(&otherCell->shared1.y, -force.y);
                } else {
                    auto force = Math::getNormalized(posDelta) * (cudaSimulationParameters.maxCollisionDistance.value - Math::length(posDelta))
                        * cudaSimulationParameters.repulsionStrength.value * fixedFactor;
                    atomicAdd(&object->shared1.x, force.x);
                    atomicAdd(&object->shared1.y, force.y);
                    atomicAdd(&otherCell->shared1.x, -force.x);
                    atomicAdd(&otherCell->shared1.y, -force.y);
                }

                //fusion
                auto cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, object->pos);
                auto cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.cellFusionVelocity, data, object->pos);

                if (object->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS && (object->sticky || otherCell->sticky)
                    && Math::length(velDelta) >= cellFusionVelocity && isApproaching && object->usableEnergy <= cellMaxBindingEnergy
                    && otherCell->usableEnergy <= cellMaxBindingEnergy && !object->fixed && !otherCell->fixed) {
                    ObjectConnectionProcessor::scheduleAddConnectionPair(data, cell, otherCell);
                }
            }
        });
    }
}

__inline__ __device__ void ObjectProcessor::checkForces(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        object->density = object->shared2.x;
        if (object->fixed) {
            continue;
        }

        if (Math::length(object->shared1) > ParameterCalculator::calcParameter(cudaSimulationParameters.maxForce, data, object->pos, object->color)) {
            if (data.primaryNumberGen.random() < cudaSimulationParameters.maxForceDecayProbability) {
                ObjectConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::applyForces(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }

        object->vel += object->shared1;
        if (Math::length(object->vel) > cudaSimulationParameters.maxVelocity.value) {
            object->vel = Math::getNormalized(object->vel) * cudaSimulationParameters.maxVelocity.value;
        }
        object->shared1 = {0, 0};
    }
}

__inline__ __device__ void ObjectProcessor::calcConnectionForces(SimulationData& data, bool calcAngularForces)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (0 == object->numConnections || object->fixed) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = object->connections[object->numConnections - 1].object->pos - object->pos;
        data.cellMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = object->stiffness * object->stiffness;

        auto numConnections = object->numConnections;
        for (int i = 0; i < numConnections; ++i) {
            auto connectedCell = object->connections[i].cell;
            auto connectedCellStiffnessSquared = connectedCell->stiffness * connectedCell->stiffness;

            auto displacement = connectedCell->pos - object->pos;
            data.cellMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = object->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::getNormalized(displacement) * deviation * (cellStiffnessSquared + connectedCellStiffnessSquared) / 6;

            if (calcAngularForces) {
                auto lastIndex = (i + numConnections - 1) % numConnections;
                auto lastConnectedCell = object->connections[lastIndex].cell;

                auto referenceAngleFromPrevious = object->connections[i].angleFromPrevious;

                auto r1 = prevDisplacement;
                auto r2 = displacement;
                Math::rotateQuarterClockwise(r1);
                Math::rotateQuarterCounterClockwise(r2);

                auto angle = Math::angleOfVector(displacement);
                auto prevAngle = Math::angleOfVector(prevDisplacement);
                auto theta = Math::getNormalizedAngle(angle - prevAngle, 0.0f);

                if (theta < referenceAngleFromPrevious) {
                    r1 *= -1.0f;
                    r2 *= -1.0f;
                }
                auto g = 5e-4f * abs(Math::getNormalizedAngle(theta - referenceAngleFromPrevious, -180.0f)) * cellStiffnessSquared;
                auto strength1 = g / max(Math::lengthSquared(r1), cudaSimulationParameters.minCellDistance.value);
                auto strength2 = g / max(Math::lengthSquared(r2), cudaSimulationParameters.minCellDistance.value);
                auto force2 = r1 * strength1;
                auto force1 = r2 * strength2;

                if (!connectedCell->fixed && !lastConnectedCell->fixed) {
                    atomicAdd(&connectedCell->shared1.x, force1.x);
                    atomicAdd(&connectedCell->shared1.y, force1.y);
                    atomicAdd(&lastConnectedCell->shared1.x, force2.x);
                    atomicAdd(&lastConnectedCell->shared1.y, force2.y);
                }
                force -= force1 + force2;
            }

            prevDisplacement = displacement;
        }
        atomicAdd(&object->shared1.x, force.x);
        atomicAdd(&object->shared1.y, force.y);
    }
}

__inline__ __device__ void ObjectProcessor::checkConnections(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }

        bool scheduleForDestruction = false;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedCell = object->connections[i].cell;

            auto displacement = connectedCell->pos - object->pos;
            data.cellMap.correctDirection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            ObjectConnectionProcessor::scheduleDeleteAllConnections(data, cell);
            for (int i = 0; i < object->numConnections; ++i) {
                auto connectedCell = object->connections[i].cell;
                connectedCell->cellState = CellState_Detaching;
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletPositionUpdate(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value;
            data.cellMap.correctPosition(object->pos);
        } else {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value
                + object->shared1 * cudaSimulationParameters.timestepSize.value * cudaSimulationParameters.timestepSize.value / 2;
            data.cellMap.correctPosition(object->pos);
            object->shared2 = object->shared1;  //forces
            object->shared1 = {0, 0};
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        auto acceleration = (object->shared1 + object->shared2) / 2;
        object->vel += acceleration * cudaSimulationParameters.timestepSize.value;
    }
}

__inline__ __device__ void ObjectProcessor::aging(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        if (object->fixed) {
            continue;
        }
        ++object->age;


        if (cudaSimulationParameters.colorTransitionRulesToggle.value) {
            int transitionDuration;
            int targetColor;
            auto color = calcMod(object->color, MAX_COLORS);
            auto index = ParameterCalculator::getFirstMatchingLayerOrBase(data, object->pos, cudaSimulationParameters.colorTransitionRules);
            if (index == -1) {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.baseValue[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.baseValue[color].targetColor;
            } else {
                transitionDuration = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].duration;
                targetColor = cudaSimulationParameters.colorTransitionRules.layerValues[index].value[color].targetColor;
            }
            if (transitionDuration > 0 && object->age > transitionDuration) {
                object->color = targetColor;
                object->age = 0;
            }
        }
        if (object->cellState == CellState_Ready && object->activationTime > 0) {
            --object->activationTime;
        }
    }
}


__inline__ __device__ void ObjectProcessor::cellStateTransition_calcFutureState(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);

        bool isSameCreatureNeighborDetaching = false;
        bool isOtherCreatureNeighborDetaching = false;
        bool isSameCreatureNeighborReviving = false;
        bool isNeighborActivating = false;
        //int activatingObjectConnection = -1;
        for (int i = 0; i < object->numConnections; ++i) {
            auto const& connectedCell = object->connections[i].cell;
            if (object->isSameCreature(connectedCell)) {
                auto connectedCellState = connectedCell->cellState;
                if (connectedCellState == CellState_Detaching) {
                    isSameCreatureNeighborDetaching = true;
                } else if (connectedCellState == CellState_Reviving) {
                    isSameCreatureNeighborReviving = true;
                } else if (connectedCellState == CellState_Activating) {
                    if (connectedCell->connections[0].cell == cell) {
                        isNeighborActivating = true;
                        //activatingObjectConnection = i;
                    }
                }
            } else {
                if (connectedCell->cellState == CellState_Detaching) {
                    isOtherCreatureNeighborDetaching = true;
                }
            }
        }

        auto origCellState = object->cellState;
        auto cellState = origCellState;

        if (object->fixed) {
            cellState = CellState_Ready;
        } else {
            if (origCellState == CellState_Activating) {
                cellState = CellState_Ready;
                if (cudaSimulationParameters.cellAgeLimiterToggle.value && cudaSimulationParameters.resetCellAgeAfterActivation.value) {
                    atomicExch(&object->age, 0);
                }
            } else if (origCellState == CellState_Reviving) {
                cellState = CellState_Ready;
            } else if (origCellState == CellState_Constructing) {
                if (isNeighborActivating) {
                    cellState = CellState_Activating;
                }
                if (isOtherCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences.value != CellDeathConsequences_None) {
                    cellState = CellState_Detaching;
                }
            } else if (origCellState == CellState_Detaching) {
                if (isSameCreatureNeighborReviving && cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_DetachedPartsDie) {
                    cellState = CellState_Reviving;
                }
                if (cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_None) {
                    cellState = CellState_Ready;
                }
            } else if (origCellState == CellState_Ready) {
                if (isSameCreatureNeighborDetaching && cudaSimulationParameters.cellDeathConsequences.value != CellDeathConsequences_None) {
                    if (cudaSimulationParameters.cellDeathConsequences.value == CellDeathConsequences_DetachedPartsDie && object->creature != nullptr
                        && object->headCell) {
                        cellState = CellState_Reviving;
                    } else {
                        cellState = CellState_Detaching;
                    }
                }
            }
            if (object->cellType == CellType_Free || object->cellType == CellType_Structure) {
                cellState = origCellState;
            }
        }
        object->tempValue.as_uint32_float.uint32Part = cellState;
    }
}

__inline__ __device__ void ObjectProcessor::cellStateTransition_applyNextState(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        object->cellState = object->tempValue.as_uint32_float.uint32Part;
        object->tempValue.as_uint32_float.uint32Part = 0;
    }
}

__inline__ __device__ void ObjectProcessor::frontAngleUpdate_calcFutureValue(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->creature == nullptr) {
            continue;
        }
        if (object->frontAngleId != object->creature->frontAngleId) {
            if (!object->headCell) {
                auto update = false;
                for (int i = 0, j = object->numConnections; i < j; ++i) {
                    auto const& otherCell = object->connections[i].cell;
                    if (!object->isSameCreature(otherCell)) {
                        continue;
                    }
                    if (otherCell->frontAngleId == object->creature->frontAngleId) {
                        auto frontAngle_otherCell_cell =
                            Math::getNormalizedAngle(otherCell->frontAngle + getInitialAngelSpan(otherCell, cell, otherCell->connections[0].cell), -180.0f);
                        auto frontAngle_cell_otherCell = Math::getNormalizedAngle(frontAngle_otherCell_cell - 180.0f, -180.0f);
                        auto frontAngle_cell_connection0 = Math::getNormalizedAngle(frontAngle_cell_otherCell + getInitialAngelSpan(cell, 0, i), -180.0f);
                        object->tempValue.as_uint32_float.floatPart = frontAngle_cell_connection0;

                        update = true;
                        break;
                    }
                }
                if (!update) {
                    object->tempValue.as_uint32_float.floatPart = FrontAngleId_NoUpdate;
                }
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::frontAngleUpdate_applyFutureValue(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->creature == nullptr) {
            continue;
        }
        if (object->frontAngleId != object->creature->frontAngleId) {
            if (object->headCell) {
                object->frontAngleId = object->creature->frontAngleId;
                object->frontAngle = object->creature->genome->frontAngle;
            } else {
                if (object->tempValue.as_uint32_float.floatPart != FrontAngleId_NoUpdate) {
                    object->frontAngleId = object->creature->frontAngleId;
                    object->frontAngle = object->tempValue.as_uint32_float.floatPart;
                }
                object->tempValue.as_uint32_float.floatPart = 0;
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::applyInnerFriction(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    auto const innerFriction = cudaSimulationParameters.innerFriction.value;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        for (int index = 0; index < object->numConnections; ++index) {
            auto connectingCell = object->connections[index].cell;
            if (connectingCell->fixed) {
                continue;
            }
            auto posDelta = object->pos - connectingCell->pos;
            auto distance = Math::length(posDelta);
            if (distance > NEAR_ZERO) {
                auto direction = posDelta / distance;
                auto velDelta = object->vel - connectingCell->vel;
                auto velDelta_part = Math::dot(velDelta, direction);

                auto delta = direction * innerFriction * velDelta_part;
                atomicAdd(&object->vel.x, -delta.x * 0.5f);
                atomicAdd(&object->vel.y, -delta.y * 0.5f);
                atomicAdd(&connectingCell->vel.x, delta.x * 0.5f);
                atomicAdd(&connectingCell->vel.y, delta.y * 0.5f);
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::applyFriction(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto const partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }

        auto friction = ParameterCalculator::calcParameter(cudaSimulationParameters.friction, data, object->pos);
        object->vel = object->vel * (1.0f - friction);
    }
}

__inline__ __device__ void ObjectProcessor::radiation(SimulationData& data)
{
    auto& cells = data.entities.objects;

    auto partition = calcSystemThreadPartition(cells.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        if (object->cellType == CellType_Structure) {
            continue;
        }
        if (data.primaryNumberGen.random() < cudaSimulationParameters.radiationProbability) {

            auto radiation1 = 0.0f;
            auto radiation2 = 0.0f;
            auto const& cellEnergy = object->usableEnergy;
            auto const& cellRawEnergy = object->rawEnergy;
            if (object->usableEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[object->color]) {
                radiation1 += cudaSimulationParameters.radiationType2_strength.value[object->color];
            }
            if (object->rawEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[object->color]) {
                radiation2 += cudaSimulationParameters.radiationType2_strength.value[object->color];
            }
            if (object->age > cudaSimulationParameters.radiationType1_minimumAge.value[object->color]) {
                radiation1 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, object->pos, object->color);
                radiation2 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, object->pos, object->color);
            }
            radiation1 *= cellEnergy;
            radiation2 *= cellRawEnergy;

            radiation1 = max(min(radiation1 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, cellEnergy - 1), 0.0f);
            radiation2 = max(min(radiation2 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, cellRawEnergy - 1), 0.0f);

            if (radiation1 > 0 || radiation2 > 0) {
                float2 particleVel = object->vel * cudaSimulationParameters.radiationVelocityMultiplier
                    + Math::unitVectorOfAngle(data.primaryNumberGen.random() * 360) * cudaSimulationParameters.radiationVelocityPerturbation;
                float2 particlePos = object->pos + Math::getNormalized(particleVel) * 1.5f
                    - particleVel;  // minus particleVel because particle will still be moved in current time step
                data.cellMap.correctPosition(particlePos);

                EnergyParticleProcessor::createEnergyParticle(data, particlePos, particleVel, object->color, radiation1 + radiation2);

                object->usableEnergy -= radiation1;
                object->rawEnergy -= radiation2;
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::decay(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->fixed) {
            continue;
        }
        auto cellMaxBindingEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.cellMaxBindingEnergy, data, object->pos);
        if (object->usableEnergy > cellMaxBindingEnergy) {
            ObjectConnectionProcessor::scheduleDeleteAllConnections(data, cell);
        }

        auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);

        if (object->cellState == CellState_Dying || object->cellState == CellState_Detaching) {
            auto cellDeathProbability = ParameterCalculator::calcParameter(cudaSimulationParameters.cellDeathProbability, data, object->pos, object->color);
            if (data.primaryNumberGen.random() <= cellDeathProbability) {
                ObjectConnectionProcessor::scheduleDeleteCell(data, index);
            }
        }

        bool cellDestruction = false;
        if (object->usableEnergy < cellMinEnergy) {
            cellDestruction = true;
        }

        auto cellMaxAge = cudaSimulationParameters.maxCellAge.value[object->color];
        if (cudaSimulationParameters.cellAgeLimiterToggle.value && object->cellType != CellType_Free && object->cellType != CellType_Structure
            && object->cellTriggered == CellTriggered_No && object->cellState == CellState_Ready && object->activationTime == 0) {
            bool adjacentCellsUsed = false;
            for (int i = 0; i < object->numConnections; ++i) {
                if (object->connections[i].object->cellTriggered == CellTriggered_Yes) {
                    adjacentCellsUsed = true;
                    break;
                }
            }
            if (!adjacentCellsUsed) {
                auto cellInactiveMaxAge = ParameterCalculator::calcParameter(cudaSimulationParameters.maxAgeForInactiveCells, data, object->pos, object->color);

                cellMaxAge = toInt(cellInactiveMaxAge);
            }
        }
        if (cudaSimulationParameters.cellAgeLimiterToggle.value && object->cellType == CellType_Free) {
            cellMaxAge = cudaSimulationParameters.freeCellMaxAge.value[object->color];
        }
        if (cellMaxAge > 0 && object->age > cellMaxAge) {
            cellDestruction = true;
        }

        if (cellDestruction) {
            auto orig = atomicExch(&object->cellState, CellState_Dying);
            if (orig != CellState_Dying) {
                for (int i = 0; i < object->numConnections; ++i) {
                    auto const& connectedCell = object->connections[i].cell;
                    auto origConnected = atomicExch(&connectedCell->cellState, CellState_Detaching);
                    if (origConnected == CellState_Dying) {
                        atomicExch(&connectedCell->cellState, CellState_Dying);
                    }
                }
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::resetDensity(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);

        object->density = 1.0f;
    }
}

__inline__ __device__ void ObjectProcessor::updateCellEvents(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->eventCounter > 0) {
            --object->eventCounter;
        }
    }
}

__inline__ __device__ void ObjectProcessor::performEnergyFlow(SimulationData& data)
{
    auto& cells = data.entities.objects;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = cells.at(index);
        if (object->numConnections == 0) {
            continue;
        }
        //if (object->cellState == CellState_Constructing || object->cellState == CellState_Activating) {
        //    continue;
        //}
        auto i = *data.timestep % object->numConnections;
        auto& connectedCell = object->connections[i].cell;

        // Flow of usable energy
        {
            auto cellMinEnergy = ParameterCalculator::calcParameter(cudaSimulationParameters.minCellEnergy, data, object->pos, object->color);
            auto cellNormalEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];

            auto needsEnergy = [](Object* cell) {
                return (object->cellState == CellState_Ready || object->cellState == CellState_Detaching || object->cellState == CellState_Reviving)
                    && object->cellType == CellType_Constructor && object->creature
                    && !ConstructorHelper::isFinished(object->cellTypeData.constructor, *object->creature->genome);
            };
            auto lowEnergy = [&](Object* cell) { return object->usableEnergy < cellNormalEnergy; };

            auto cellNeedsEnergy = needsEnergy(cell);
            auto connectedCellNeedsEnergy = needsEnergy(connectedCell);
            auto connectedCellLowEnergy = lowEnergy(cell);

            auto flow = 0.0f;
            if (connectedCellLowEnergy) {
                if (object->usableEnergy > connectedCell->usableEnergy) {
                    flow = (object->usableEnergy - connectedCell->usableEnergy) / 2;
                }
            } else {
                if ((cellNeedsEnergy && connectedCellNeedsEnergy) || (!cellNeedsEnergy && !connectedCellNeedsEnergy)) {
                    if (object->usableEnergy > connectedCell->usableEnergy) {
                        flow = (object->usableEnergy - connectedCell->usableEnergy) / 2;
                    }
                }
                if (!cellNeedsEnergy && connectedCellNeedsEnergy) {
                    if (object->usableEnergy > cellNormalEnergy) {
                        flow = object->usableEnergy - cellNormalEnergy;
                    }
                }
            }

            if (flow > 0) {
                flow = min(2.0f, flow);
                auto orig = atomicAdd(&object->usableEnergy, -flow);
                if (orig < cellMinEnergy) {
                    atomicAdd(&object->usableEnergy, flow);
                } else {
                    atomicAdd(&connectedCell->usableEnergy, flow);
                }
            }
        }

        // Flow of raw energy
        {
            if ((object->cellState == CellState_Ready || object->cellState == CellState_Detaching || object->cellState == CellState_Reviving)
                && (connectedCell->cellState == CellState_Ready || connectedCell->cellState == CellState_Detaching
                    || connectedCell->cellState == CellState_Reviving)
                && connectedCell->rawEnergy < SimulationParameters::maxRawEnergyThresholdForConduction) {

                auto flow = 0.0f;
                auto maxFlow = 0.0f;
                if (connectedCell->cellType == CellType_Digestor) {
                    maxFlow += connectedCell->cellTypeData.digestor.rawEnergyConductivity
                        * cudaSimulationParameters.maxRawEnergyConductivity.value[connectedCell->color];
                    if (object->cellType == CellType_Digestor) {
                        maxFlow += object->cellTypeData.digestor.rawEnergyConductivity * cudaSimulationParameters.maxRawEnergyConductivity.value[object->color];

                        auto cellConversionRate = 1.0f - object->cellTypeData.digestor.rawEnergyConductivity;
                        auto connectedCellConversionRate = 1.0f - connectedCell->cellTypeData.digestor.rawEnergyConductivity;
                        if (cellConversionRate == 0 && connectedCellConversionRate == 0) {
                            cellConversionRate = 1;
                            connectedCellConversionRate = 1;
                        }

                        auto targetEnergy =
                            (object->rawEnergy + connectedCell->rawEnergy) * cellConversionRate / (cellConversionRate + connectedCellConversionRate);
                        flow = max(object->rawEnergy - targetEnergy, 0.0f);
                    } else {
                        flow = object->rawEnergy;
                    }
                }

                flow = min(maxFlow, flow);
                auto orig = atomicAdd(&object->rawEnergy, -flow);
                if (orig < 0) {
                    atomicAdd(&object->rawEnergy, flow);
                } else {
                    atomicAdd(&connectedCell->rawEnergy, flow);
                }
            }
        }
    }
}

__device__ __inline__ float ObjectProcessor::getInitialAngelSpan(Object* cell, int connectionIndex1, int connectionIndex2)
{
    if ((connectionIndex1 - connectionIndex2 + object->numConnections) % object->numConnections == 0) {
        return 0;
    }
    auto result = 0.0f;
    for (int i = connectionIndex1 + 1; i < connectionIndex1 + object->numConnections; i++) {
        auto index = i % object->numConnections;
        result += MuscleProcessor::getInitialAngleFromPrevious(cell, index);
        if (index == connectionIndex2) {
            break;
        }
    }
    return Math::getNormalizedAngle(result, -180.0f);
}

__device__ __inline__ float ObjectProcessor::getInitialAngelSpan(Object* cell, Object* connectedCell1, Object* connectedCell2)
{
    auto connectionIndex1 = -1;
    auto connectionIndex2 = -1;
    for (int i = 0; i < object->numConnections; i++) {
        if (object->connections[i].cell == connectedCell1) {
            connectionIndex1 = i;
        }
        if (object->connections[i].cell == connectedCell2) {
            connectionIndex2 = i;
        }
    }
    if (connectionIndex1 == -1 || connectionIndex2 == -1) {
        return 0;
    }
    return getInitialAngelSpan(cell, connectionIndex1, connectionIndex2);
}
