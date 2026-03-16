#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <EngineInterface/CellTypeConstants.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "ConstructorHelper.cuh"
#include "EnergyProcessor.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "MuscleProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ParameterCalculator.cuh"
#include "Physics.cuh"
#include "TOs.cuh"

namespace cg = cooperative_groups;

class ObjectProcessor
{
public:
    __inline__ __device__ static void init(SimulationData& data);
    __inline__ __device__ static void updateMap(SimulationData& data);
    __inline__ __device__ static void clearDensityMap(SimulationData& data);
    __inline__ __device__ static void fillDensityMap(SimulationData& data);

    __inline__ __device__ static void calcFluidForces_reconnectCells_correctOverlap(SimulationData& data);
    __inline__ __device__ static void calcFluidBoundaryForces(SimulationData& data);
    __inline__ __device__ static void checkForces(SimulationData& data);
    __inline__ __device__ static void applyForces(SimulationData& data);  //prerequisite: data from calcCollisions_reconnectCells_correctOverlap

    __inline__ __device__ static void calcConnectionForces(SimulationData& data, bool calcAngularForces);
    __inline__ __device__ static void checkConnections(SimulationData& data);
    __inline__ __device__ static void verletPositionUpdate(SimulationData& data);
    __inline__ __device__ static void verletVelocityUpdate(SimulationData& data);

    __inline__ __device__ static void applyInnerFriction(SimulationData& data);
    __inline__ __device__ static void applyFriction(SimulationData& data);

    __inline__ __device__ static void radiation(SimulationData& data);

    __inline__ __device__ static void resetDensity(SimulationData& data);

private:
    static auto constexpr MaxBarrierCellsForCollision = 10;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ObjectProcessor::init(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        object->shared1 = {0, 0};
        object->nextObject = nullptr;
        object->tempValue.as_uint64 = 0;
    }
}

__inline__ __device__ void ObjectProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcBlockPartition(data.entities.objects.getNumEntries());
    Object** objectPointers = &data.entities.objects.at(partition.startIndex);
    data.objectMap.set_block(partition.numElements(), objectPointers);
}

__inline__ __device__ void ObjectProcessor::clearDensityMap(SimulationData& data)
{
    data.preprocessedSimulationData.densityMap.clear();
}

__inline__ __device__ void ObjectProcessor::fillDensityMap(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto object = data.entities.objects.at(index);
        if (object->type == ObjectType_FreeCell) {
            data.preprocessedSimulationData.densityMap.addFreeCell(object);
        } else if (object->type == ObjectType_Structure) {
            data.preprocessedSimulationData.densityMap.addStructureObject(object);
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

    auto& objects = data.entities.objects;
    auto blockPartition = calcBlockPartition(objects.getNumEntries());
    auto const& smoothingLength_base = cudaSimulationParameters.smoothingLength.value;

    for (int objectIndex = blockPartition.startIndex; objectIndex <= blockPartition.endIndex; ++objectIndex) {
        auto& object = objects.at(objectIndex);
        auto smoothingLength = smoothingLength_base;
        auto isObjectFluid = object->type == ObjectType_Fluid;
        if (isObjectFluid) {
            smoothingLength *= 2.0f;  // Use larger smoothing length for fluids
        }

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
            cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.objectFusionVelocity, data, object->pos);

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

        for (int scanIndex = toInt(block.thread_rank()); scanIndex < scanLength * scanLength; scanIndex += block.size()) {
            int2 scanPos{cellPosInt.x + (scanIndex % scanLength), cellPosInt.y + (scanIndex / scanLength)};
            data.objectMap.correctPosition(scanPos);
            auto otherObject = data.objectMap.getFirst(scanPos);
            for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
                if (!otherObject) {
                    break;
                }
                if ((isObjectFluid && otherObject->type == ObjectType_Fluid) || (!isObjectFluid && otherObject->type != ObjectType_Fluid)) {
                    auto posDelta = object->pos - otherObject->pos;

                    data.objectMap.correctDirection(posDelta);
                    auto adaptedDistance = Math::length(posDelta);
                    auto origDistance = adaptedDistance;
                    if ((object->numConnections < 3 || otherObject->numConnections < 3) && object->type == ObjectType_Cell
                        && otherObject->type == ObjectType_Cell && object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                        adaptedDistance *= 2.0f;  // Reduce range of cell repulsion within creature by scaling distance
                    }

                    if (otherObject->fixed && adaptedDistance <= smoothingLength * 2 && object->detached + otherObject->detached != 1) {
                        auto index = atomicAdd(&numFixedCells, 1);
                        if (index < MaxBarrierCellsForCollision) {
                            fixedCells[index] = otherObject;
                        }
                    }

                    if (!otherObject->fixed && adaptedDistance <= smoothingLength * 2 && object->detached + otherObject->detached != 1) {

                        // Calc density
                        auto otherMass = otherObject->getMassForSPH();
                        localDensity += otherMass * calcKernel(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength);

                        if (object != otherObject) {

                            // Overlap correction
                            if (!object->fixed && origDistance < cudaSimulationParameters.minObjectDistance.value) {
                                localCellPosDelta.x += posDelta.x * cudaSimulationParameters.minObjectDistance.value / 5;
                                localCellPosDelta.y += posDelta.y * cudaSimulationParameters.minObjectDistance.value / 5;
                            }

                            auto velDelta = object->vel - otherObject->vel;
                            bool isConnected = false;
                            for (int i = 0; i < object->numConnections; ++i) {
                                auto const& connectedObject = object->connections[i].object;
                                if (connectedObject == otherObject) {
                                    isConnected = true;
                                }
                            }
                            if (!isConnected) {

                                // Calc forces: for simplicity pressure = density
                                auto const& cellPressure = object->density;              // Optimization: using the density from last time step
                                auto const& otherObjectPressure = otherObject->density;  // Optimization: using the density from last time step
                                auto factor =
                                    cellPressure / (object->density * object->density) + otherObjectPressure / (otherObject->density * otherObject->density);

                                if (adaptedDistance > NEAR_ZERO) {
                                    float kernel_d = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                                    auto F_pressureDelta = posDelta / (-adaptedDistance) * factor * kernel_d * otherMass;
                                    localF_pressure.x += F_pressureDelta.x;
                                    localF_pressure.y += F_pressureDelta.y;

                                    auto F_viscosityDelta =
                                        velDelta / otherObject->density * adaptedDistance * kernel_d / (adaptedDistance * adaptedDistance + 0.25f) * otherMass;
                                    localF_viscosity.x += F_viscosityDelta.x;
                                    localF_viscosity.y += F_viscosityDelta.y;
                                }
                            }

                            // Fusion
                            if (Math::length(velDelta) >= cellFusionVelocity && object->numConnections < MAX_OBJECT_CONNECTIONS
                                && otherObject->numConnections < MAX_OBJECT_CONNECTIONS && (object->sticky || otherObject->sticky) && !object->fixed
                                && !otherObject->fixed) {
                                ObjectConnectionProcessor::scheduleAddConnectionPair(data, object, otherObject);
                            }
                        }
                    }
                }
                otherObject = otherObject->nextObject;
            }
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
                    auto distance = data.objectMap.getDistance(object->pos, fixedCell->pos);
                    if (!closestFixedCell || distance < closestFixedCellDistance) {
                        closestFixedCell = fixedCell;
                        closestFixedCellDistance = distance;
                    }
                }

                float2 r{0, 0};
                if (closestFixedCell->numConnections <= 1) {
                    r = data.objectMap.getCorrectedDirection(object->pos - closestFixedCell->pos);
                } else {
                    auto angleToCell = Math::angleOfVector(data.objectMap.getCorrectedDirection(object->pos - closestFixedCell->pos));
                    auto numConnections = closestFixedCell->numConnections;
                    for (int i = 0; i < numConnections; ++i) {
                        auto otherObject1 = closestFixedCell->connections[i].object;
                        auto otherObject2 = closestFixedCell->connections[(i + 1) % numConnections].object;
                        auto angleToOtherObject1 = Math::angleOfVector(data.objectMap.getCorrectedDirection(otherObject1->pos - closestFixedCell->pos));
                        auto angleToOtherObject2 = Math::angleOfVector(data.objectMap.getCorrectedDirection(otherObject2->pos - closestFixedCell->pos));
                        if (Math::isAngleInBetween(angleToOtherObject1, angleToOtherObject2, angleToCell)) {
                            r = otherObject2->pos - otherObject1->pos;
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
            object->shared1 +=
                F_pressure * cudaSimulationParameters.pressureStrength.value * density + F_viscosity * cudaSimulationParameters.viscosityStrength.value;
            object->shared2.x = density;
        }
        block.sync();
    }
}

__inline__ __device__ void ObjectProcessor::calcFluidBoundaryForces(SimulationData& data)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto& objects = data.entities.objects;
    auto blockPartition = calcBlockPartition(objects.getNumEntries());
    auto const smoothingLength = cudaSimulationParameters.smoothingLength.value * 2.0f;  // Fluid uses 2x base smoothing length

    for (int objectIndex = blockPartition.startIndex; objectIndex <= blockPartition.endIndex; ++objectIndex) {
        auto& object = objects.at(objectIndex);

        __shared__ int scanLength;
        __shared__ int2 cellPosInt;
        __shared__ float2 F_boundary;

        if (block.thread_rank() == 0) {
            F_boundary = {0, 0};
            if (object->type == ObjectType_Fluid) {
                int radiusInt = ceilf(smoothingLength * 2);
                scanLength = radiusInt * 2 + 1;
                cellPosInt = {floorInt(object->pos.x) - radiusInt, floorInt(object->pos.y) - radiusInt};
            } else {
                scanLength = 0;
            }
        }
        block.sync();

        float2 localF_boundary = {0, 0};

        for (int scanIndex = toInt(block.thread_rank()); scanIndex < scanLength * scanLength; scanIndex += block.size()) {
            int2 scanPos{cellPosInt.x + (scanIndex % scanLength), cellPosInt.y + (scanIndex / scanLength)};
            data.objectMap.correctPosition(scanPos);
            auto otherObject = data.objectMap.getFirst(scanPos);
            for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
                if (!otherObject) {
                    break;
                }
                if (otherObject->type != ObjectType_Fluid && otherObject != object && object->detached + otherObject->detached != 1) {

                    auto posDelta = object->pos - otherObject->pos;
                    data.objectMap.correctDirection(posDelta);
                    auto adaptedDistance = Math::length(posDelta);

                    if (adaptedDistance <= smoothingLength * 2 && adaptedDistance > NEAR_ZERO) {
                        auto solidMass = otherObject->getMassForSPH();

                        float kernel_d_val = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                        // Repulsion force on fluid from solid boundary.
                        // Factor 2/rho_f mirrors the symmetric SPH pressure factor (1/rho_f + 1/rho_f)
                        // and is proportional to solid mass so that a heavier boundary repels more strongly.
                        auto F_on_fluid = posDelta / (-adaptedDistance) * (2.0f / max(NEAR_ZERO, object->density)) * kernel_d_val * solidMass * 0.3f;
                        localF_boundary += F_on_fluid;

                        // Counter-force on solid: equal and opposite (Newton's 3rd law).
                        // pressureStrength is applied here directly since this force bypasses the
                        // block-local F_boundary accumulation path (which gets pressureStrength at the end).
                        atomicAdd(&otherObject->shared1.x, -F_on_fluid.x * cudaSimulationParameters.pressureStrength.value);
                        atomicAdd(&otherObject->shared1.y, -F_on_fluid.y * cudaSimulationParameters.pressureStrength.value);
                    }
                }
                otherObject = otherObject->nextObject;
            }
        }

        // Warp-level reduction followed by atomic accumulation across warps
        float sumFx = cg::reduce(warp, localF_boundary.x, cg::plus<float>());
        float sumFy = cg::reduce(warp, localF_boundary.y, cg::plus<float>());
        if (warp.thread_rank() == 0) {
            atomicAdd_block(&F_boundary.x, sumFx);
            atomicAdd_block(&F_boundary.y, sumFy);
        }
        block.sync();

        if (block.thread_rank() == 0) {
            object->shared1 += F_boundary * cudaSimulationParameters.pressureStrength.value;
        }
        block.sync();
    }
}

__inline__ __device__ void ObjectProcessor::checkForces(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        object->density = object->shared2.x;
        if (object->fixed) {
            continue;
        }

        if (Math::length(object->shared1) > ParameterCalculator::calcParameter(cudaSimulationParameters.maxForce, data, object->pos, object->color)) {
            if (data.primaryNumberGen.random() < cudaSimulationParameters.maxForceDecayProbability) {
                ObjectConnectionProcessor::scheduleDeleteAllConnections(data, object);
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::applyForces(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        auto acceleration = object->shared1 / max(0.05f, object->density) * 0.5f;
        if (Math::length(acceleration) > cudaSimulationParameters.maxAcceleration) {
            acceleration = Math::getNormalized(acceleration) * cudaSimulationParameters.maxAcceleration;
        }
        object->vel += acceleration;
        if (Math::length(object->vel) > cudaSimulationParameters.maxVelocity.value) {
            object->vel = Math::getNormalized(object->vel) * cudaSimulationParameters.maxVelocity.value;
        }
        object->shared1 = {0, 0};
    }
}

__inline__ __device__ void ObjectProcessor::calcConnectionForces(SimulationData& data, bool calcAngularForces)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (0 == object->numConnections || object->fixed) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = object->connections[object->numConnections - 1].object->pos - object->pos;
        data.objectMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = object->stiffness * object->stiffness;

        auto numConnections = object->numConnections;
        for (int i = 0; i < numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            auto connectedObjectStiffnessSquared = connectedObject->stiffness * connectedObject->stiffness;

            auto displacement = connectedObject->pos - object->pos;
            data.objectMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = object->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            force = force + Math::getNormalized(displacement) * deviation * (cellStiffnessSquared + connectedObjectStiffnessSquared) / 6;

            if (calcAngularForces) {
                auto lastIndex = (i + numConnections - 1) % numConnections;
                auto lastConnectedObject = object->connections[lastIndex].object;

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
                auto strength1 = g / max(Math::lengthSquared(r1), 0.7f);
                auto strength2 = g / max(Math::lengthSquared(r2), 0.7f);
                auto force2 = r1 * strength1;
                auto force1 = r2 * strength2;

                if (!connectedObject->fixed && !lastConnectedObject->fixed) {
                    atomicAdd(&connectedObject->shared1.x, force1.x);
                    atomicAdd(&connectedObject->shared1.y, force1.y);
                    atomicAdd(&lastConnectedObject->shared1.x, force2.x);
                    atomicAdd(&lastConnectedObject->shared1.y, force2.y);
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
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }

        bool scheduleForDestruction = false;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;

            auto displacement = connectedObject->pos - object->pos;
            data.objectMap.correctDirection(displacement);
            auto actualDistance = Math::length(displacement);
            if (actualDistance > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                scheduleForDestruction = true;
            }
        }
        if (scheduleForDestruction) {
            ObjectConnectionProcessor::scheduleDeleteAllConnections(data, object);
            for (int i = 0; i < object->numConnections; ++i) {
                auto connectedObject = object->connections[i].object;
                connectedObject->typeData.cell.cellState = CellState_Detaching;
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletPositionUpdate(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value;
            data.objectMap.correctPosition(object->pos);
        } else {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value
                + object->shared1 * cudaSimulationParameters.timestepSize.value * cudaSimulationParameters.timestepSize.value / 2;
            data.objectMap.correctPosition(object->pos);
            object->shared2 = object->shared1;  //forces
            object->shared1 = {0, 0};
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        auto acceleration = (object->shared1 + object->shared2) / 2;
        object->vel += acceleration * cudaSimulationParameters.timestepSize.value;
    }
}

__inline__ __device__ void ObjectProcessor::applyInnerFriction(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    auto const innerFriction = cudaSimulationParameters.innerFriction.value;
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        for (int index = 0; index < object->numConnections; ++index) {
            auto connectedObject = object->connections[index].object;
            if (connectedObject->fixed) {
                continue;
            }
            auto posDelta = object->pos - connectedObject->pos;
            auto distance = Math::length(posDelta);
            if (distance > NEAR_ZERO) {
                auto direction = posDelta / distance;
                auto velDelta = object->vel - connectedObject->vel;
                auto velDelta_part = Math::dot(velDelta, direction);

                auto delta = direction * innerFriction * velDelta_part;
                atomicAdd(&object->vel.x, -delta.x * 0.5f);
                atomicAdd(&object->vel.y, -delta.y * 0.5f);
                atomicAdd(&connectedObject->vel.x, delta.x * 0.5f);
                atomicAdd(&connectedObject->vel.y, delta.y * 0.5f);
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::applyFriction(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }

        auto friction = ParameterCalculator::calcParameter(cudaSimulationParameters.friction, data, object->pos);
        object->vel = object->vel * (1.0f - friction);
    }
}

__inline__ __device__ void ObjectProcessor::radiation(SimulationData& data)
{
    auto& objects = data.entities.objects;

    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->fixed) {
            continue;
        }
        if (object->type == ObjectType_Structure || object->type == ObjectType_Fluid) {
            continue;
        }
        if (data.primaryNumberGen.random() < cudaSimulationParameters.radiationProbability) {

            auto radiation1 = 0.0f;
            auto radiation2 = 0.0f;
            auto usableEnergy = 0.0f;
            auto rawEnergy = 0.0f;
            auto age = 0u;

            // Fill radiation values based on object type
            if (object->type == ObjectType_Cell) {
                usableEnergy = object->typeData.cell.usableEnergy;
                rawEnergy = object->typeData.cell.rawEnergy;
                age = object->typeData.cell.age;
            } else if (object->type == ObjectType_FreeCell) {
                rawEnergy = object->typeData.freeCell.energy;
                age = object->typeData.freeCell.age;
            }

            if (usableEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[object->color]) {
                radiation1 += cudaSimulationParameters.radiationType2_strength.value[object->color];
            }
            if (rawEnergy > cudaSimulationParameters.radiationType2_energyThreshold.value[object->color]) {
                radiation2 += cudaSimulationParameters.radiationType2_strength.value[object->color];
            }
            if (age > cudaSimulationParameters.radiationType1_minimumAge.value[object->color]) {
                radiation1 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, object->pos, object->color);
                radiation2 += ParameterCalculator::calcParameter(cudaSimulationParameters.radiationType1_strength, data, object->pos, object->color);
            }
            radiation1 *= usableEnergy;
            radiation2 *= rawEnergy;

            radiation1 = max(min(radiation1 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, usableEnergy - 1), 0.0f);
            radiation2 = max(min(radiation2 / cudaSimulationParameters.radiationProbability * data.primaryNumberGen.random() * 2, rawEnergy - 1), 0.0f);

            // Radiate (same code for both cases)
            if (radiation1 > 0 || radiation2 > 0) {
                float2 particleVel = object->vel * cudaSimulationParameters.radiationVelocityMultiplier
                    + Math::unitVectorOfAngle(data.primaryNumberGen.random() * 360) * cudaSimulationParameters.radiationVelocityPerturbation;
                float2 particlePos = object->pos + Math::getNormalized(particleVel) * 1.5f
                    - particleVel;  // minus particleVel because particle will still be moved in current time step
                data.objectMap.correctPosition(particlePos);

                EnergyProcessor::createEnergyParticle(data, particlePos, particleVel, object->color, radiation1 + radiation2);

                // Update energy based on object type
                if (object->type == ObjectType_Cell) {
                    object->typeData.cell.usableEnergy -= radiation1;
                    object->typeData.cell.rawEnergy -= radiation2;
                } else if (object->type == ObjectType_FreeCell) {
                    object->typeData.freeCell.energy -= radiation2;
                }
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::resetDensity(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        object->density = 1.0f;
    }
}
