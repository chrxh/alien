#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <EngineInterface/CellTypeConstants.h>

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "ConstructorHelper.cuh"
#include "FluidKernelProfiler.cuh"
#include "ObjectConnectionProcessor.cuh"

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
    __inline__ __device__ static void applyForces(SimulationData& data);  // Prerequisite: data from calcCollisions_reconnectCells_correctOverlap

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

        data.objectMap.resetRecordLink(index);
        object->tempValue1.as_uint64 = 0;
    }
}

__inline__ __device__ void ObjectProcessor::updateMap(SimulationData& data)
{
    auto const partition = calcBlockPartition(data.entities.objects.getNumEntries());
    Object** objectPointers = &data.entities.objects.at(partition.startIndex);
    data.objectMap.set_block(partition.startIndex, partition.numElements(), objectPointers);
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
        } else if (object->type == ObjectType_Solid) {
            data.preprocessedSimulationData.densityMap.addSolidObject(object);
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

    // An object is registered in the map cell that contains its position, so all objects of a cell lie within
    // its unit square. If that square is entirely farther away than the interaction cutoff, the cell cannot
    // contribute and neither its map lookup nor its object chain has to be touched. The scan rectangle is a
    // square around a circular interaction range, so this skips its corners.
    __inline__ __device__ bool isCellInRange(float2 const& pos, int cellPosX, int cellPosY, float cutoffSquared)
    {
        auto deltaX = fmaxf(fmaxf(toFloat(cellPosX) - pos.x, pos.x - toFloat(cellPosX + 1)), 0.0f);
        auto deltaY = fmaxf(fmaxf(toFloat(cellPosY) - pos.y, pos.y - toFloat(cellPosY + 1)), 0.0f);
        return deltaX * deltaX + deltaY * deltaY <= cutoffSquared;
    }

    // Splits the linear scan index into a row and a column without an integer division, which the GPU
    // emulates in software. The rounding of the float division is corrected by the following comparisons.
    __inline__ __device__ int2 calcScanPos(int2 const& scanOrigin, int scanIndex, int scanLength, float invScanLength)
    {
        auto row = toInt(toFloat(scanIndex) * invScanLength);
        auto column = scanIndex - row * scanLength;
        if (column < 0) {
            --row;
            column += scanLength;
        } else if (column >= scanLength) {
            ++row;
            column -= scanLength;
        }
        return {scanOrigin.x + column, scanOrigin.y + row};
    }
}

__inline__ __device__ void ObjectProcessor::calcFluidForces_reconnectCells_correctOverlap(SimulationData& data)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    auto const warpIndexInBlock = toInt(block.thread_rank()) / WARP_SIZE;

    // The only state the lanes of a warp have to share: barrier objects are found by whichever lane happens to scan
    // their map cell. Everything else each lane derives for itself, which is cheaper than a slot plus a barrier.
    __shared__ Object* fixedCells[FLUID_KERNEL_WARPS][MaxBarrierCellsForCollision];
    __shared__ int numFixedObjects[FLUID_KERNEL_WARPS];

    auto& objects = data.entities.objects;
    auto const warpPartition = calcWarpPartition(objects.getNumEntries());
    auto const& smoothingLength_base = cudaSimulationParameters.smoothingLength.value;

    auto const profiling = cudaFluidKernelProfilingEnabled != 0;
    __shared__ unsigned long long profileScanCells;
    __shared__ unsigned long long profileRecords;
    unsigned long long profileInitCycles = 0;
    unsigned long long profileScanCycles = 0;
    unsigned long long profileSyncCycles = 0;
    unsigned long long profileTailCycles = 0;
    unsigned long long profileObjects = 0;
    unsigned long long profileFirstObjectCycles = 0;
    unsigned long long profileLoopCycles = 0;
    unsigned long long threadScanCells = 0;
    unsigned long long threadRecords = 0;
    auto const profileBlockStartNs = profiling ? readGlobalTimer() : 0;
    auto const profileWarpStartCycles = profiling ? readSmClock() : 0;
    if (profiling && block.thread_rank() == 0) {
        profileScanCells = 0;
        profileRecords = 0;
    }
    auto profileMark = profileWarpStartCycles;

    for (int objectIndex = warpPartition.startIndex; objectIndex <= warpPartition.endIndex; ++objectIndex) {
        auto const profileObjectStartCycles = profileMark;
        auto& object = objects.at(objectIndex);
        auto smoothingLength = smoothingLength_base;
        auto isObjectFluid = object->type == ObjectType_Fluid;
        if (isObjectFluid) {
            smoothingLength *= 2.0f;  // Use larger smoothing length for fluids
        }

        auto const cellFusionVelocity = ParameterCalculator::calcParameter(cudaSimulationParameters.objectFusionVelocity, data, object->pos);

        int const radiusInt = ceilf(smoothingLength * 2);
        int const scanLength = radiusInt * 2 + 1;
        int2 const cellPosInt = {floorInt(object->pos.x) - radiusInt, floorInt(object->pos.y) - radiusInt};

        if (warp.thread_rank() == 0) {
            numFixedObjects[warpIndexInBlock] = 0;
        }
        warp.sync();
        if (profiling) {
            auto const now = readSmClock();
            profileInitCycles += now - profileMark;
            profileMark = now;
            ++profileObjects;
        }

        // Per-thread accumulators
        float2 localF_pressure = {0, 0};
        float2 localF_viscosity = {0, 0};
        float2 localCellPosDelta = {0, 0};
        float localDensity = 0;

        auto const objectPos = object->pos;
        auto const cutoffSquared = smoothingLength * smoothingLength * 4;

        auto const invScanLength = 1.0f / toFloat(scanLength);

        auto records = data.objectMap.getRecords();
        for (int scanIndex = toInt(warp.thread_rank()); scanIndex < scanLength * scanLength; scanIndex += warp.size()) {
            int2 scanPos = calcScanPos(cellPosInt, scanIndex, scanLength, invScanLength);
            if (!isCellInRange(objectPos, scanPos.x, scanPos.y, cutoffSquared)) {
                continue;
            }
            ++threadScanCells;
            data.objectMap.correctPosition(scanPos);
            int otherIndex = data.objectMap.getFirstIndex(scanPos);
            for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
                if (otherIndex < 0) {
                    break;
                }
                ++threadRecords;
                auto const& other = &records[otherIndex];
                if ((isObjectFluid && other->type == ObjectType_Fluid) || (!isObjectFluid && other->type != ObjectType_Fluid)) {
                    auto posDelta = object->pos - other->pos;

                    data.objectMap.correctDirection(posDelta);
                    auto adaptedDistance = Math::length(posDelta);
                    auto origDistance = adaptedDistance;
                    if ((object->numConnections < 3 || other->numConnections < 3) && object->type == ObjectType_Cell && other->type == ObjectType_Cell
                        && object->typeData.cell.isSameCreature(&other->self->typeData.cell)
                        && object->typeData.cell.parentNodeIndex != other->self->typeData.cell.parentNodeIndex) {
                        adaptedDistance *= 2.0f;  // Reduce range of cell repulsion within creature by scaling distance
                    }

                    if (other->isStatic() && adaptedDistance <= smoothingLength * 2 && object->detached() + other->detached() != 1) {
                        auto index = atomicAdd(&numFixedObjects[warpIndexInBlock], 1);
                        if (index < MaxBarrierCellsForCollision) {
                            fixedCells[warpIndexInBlock][index] = other->self;
                        }
                    }

                    if (!other->isStatic() && adaptedDistance <= smoothingLength * 2 && object->detached() + other->detached() != 1) {

                        // Calc density
                        auto otherMass = getMassForSPH(other);
                        localDensity += otherMass * calcKernel(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength);

                        if (object != other->self) {

                            // Overlap correction
                            if (!object->isStatic() && origDistance < cudaSimulationParameters.minObjectDistance.value) {
                                localCellPosDelta.x += posDelta.x * cudaSimulationParameters.minObjectDistance.value / 5;
                                localCellPosDelta.y += posDelta.y * cudaSimulationParameters.minObjectDistance.value / 5;
                            }

                            auto velDelta = object->vel - other->vel;
                            bool isConnected = false;
                            for (int i = 0; i < object->numConnections; ++i) {
                                auto const& connectedObject = object->connections[i].object;
                                if (connectedObject == other->self) {
                                    isConnected = true;
                                }
                            }
                            if (!isConnected) {

                                // Calc forces: for simplicity pressure = density
                                auto const& cellPressure = object->density;        // Optimization: using the density from last time step
                                auto const& otherObjectPressure = other->density;  // Optimization: using the density from last time step
                                auto factor = cellPressure / (object->density * object->density) + otherObjectPressure / (other->density * other->density);

                                if (adaptedDistance > NEAR_ZERO) {
                                    float kernel_d = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                                    auto F_pressureDelta = posDelta / (-adaptedDistance) * factor * kernel_d * otherMass;
                                    localF_pressure.x += F_pressureDelta.x;
                                    localF_pressure.y += F_pressureDelta.y;

                                    auto F_viscosityDelta =
                                        velDelta / other->density * adaptedDistance * kernel_d / (adaptedDistance * adaptedDistance + 0.25f) * otherMass;
                                    localF_viscosity.x += F_viscosityDelta.x;
                                    localF_viscosity.y += F_viscosityDelta.y;
                                }
                            }

                            // Fusion
                            if (Math::length(velDelta) >= cellFusionVelocity && object->numConnections < MAX_OBJECT_CONNECTIONS
                                && other->numConnections < MAX_OBJECT_CONNECTIONS && (object->isSticky() || other->isSticky()) && !object->isStatic()
                                && !other->isStatic()) {
                                ObjectConnectionProcessor::scheduleAddConnectionPair(data, object, other->self);
                            }
                        }
                    }
                }
                otherIndex = other->nextObjectIndex;
            }
        }

        if (profiling) {
            auto const now = readSmClock();
            profileScanCycles += now - profileMark;
            profileMark = now;
        }

        // The warp owns the object, so its reduction already yields the total; no accumulation across warps is needed.
        float2 const F_pressure{cg::reduce(warp, localF_pressure.x, cg::plus<float>()), cg::reduce(warp, localF_pressure.y, cg::plus<float>())};
        float2 const F_viscosity{cg::reduce(warp, localF_viscosity.x, cg::plus<float>()), cg::reduce(warp, localF_viscosity.y, cg::plus<float>())};
        float2 const cellPosDelta{cg::reduce(warp, localCellPosDelta.x, cg::plus<float>()), cg::reduce(warp, localCellPosDelta.y, cg::plus<float>())};
        float const density = cg::reduce(warp, localDensity, cg::plus<float>());

        warp.sync();
        if (profiling) {
            auto const now = readSmClock();
            profileSyncCycles += now - profileMark;
            profileMark = now;
        }

        // Calculate forces with fixed objects
        if (warp.thread_rank() == 0) {
            auto const numFixedObjectsOfWarp = min(MaxBarrierCellsForCollision, numFixedObjects[warpIndexInBlock]);
            if (numFixedObjectsOfWarp > 0) {

                // Calc forces only to the closest fixed object
                Object* closestFixedObject = nullptr;
                float closestFixedObjectDistance;
                for (int i = 0; i < numFixedObjectsOfWarp; ++i) {
                    auto const& fixedCell = fixedCells[warpIndexInBlock][i];
                    auto distance = data.objectMap.getDistance(object->pos, fixedCell->pos);
                    if (!closestFixedObject || distance < closestFixedObjectDistance) {
                        closestFixedObject = fixedCell;
                        closestFixedObjectDistance = distance;
                    }
                }
                auto connectedToObject = false;
                auto numConnections = closestFixedObject->numConnections;
                for (int i = 0; i < numConnections; ++i) {
                    if (closestFixedObject->connections[i].object == object) {
                        connectedToObject = true;
                        break;
                    }
                }

                if (!connectedToObject) {
                    float2 r{0, 0};
                    if (closestFixedObject->numConnections <= 1) {
                        r = data.objectMap.getCorrectedDirection(object->pos - closestFixedObject->pos);
                    } else {
                        auto angleToObject = Math::angleOfVector(data.objectMap.getCorrectedDirection(object->pos - closestFixedObject->pos));
                        for (int i = 0; i < numConnections; ++i) {
                            auto otherObject1 = closestFixedObject->connections[i].object;
                            auto otherObject2 = closestFixedObject->connections[(i + 1) % numConnections].object;
                            auto angleToOtherObject1 = Math::angleOfVector(data.objectMap.getCorrectedDirection(otherObject1->pos - closestFixedObject->pos));
                            auto angleToOtherObject2 = Math::angleOfVector(data.objectMap.getCorrectedDirection(otherObject2->pos - closestFixedObject->pos));
                            if (Math::isAngleInBetween(angleToOtherObject1, angleToOtherObject2, angleToObject)) {
                                r = otherObject2->pos - otherObject1->pos;
                                Math::rotateQuarterCounterClockwise(r);
                                break;
                            }
                        }
                    }
                    auto vr = object->vel - closestFixedObject->vel;
                    auto dot_vr_r = Math::dot(vr, r);

                    if (dot_vr_r < 0) {
                        auto truncated_r_squared = max(0.05f, Math::lengthSquared(r));
                        auto truncated_distance = max(0.05f, closestFixedObjectDistance);
                        object->tempValue1.as_float2 +=
                            (vr - r * 2 * dot_vr_r / truncated_r_squared + closestFixedObject->vel - object->vel) / truncated_distance;
                    }
                }
            }

            object->pos += cellPosDelta;
            object->tempValue1.as_float2 +=
                (F_pressure * cudaSimulationParameters.pressureStrength.value * density + F_viscosity * cudaSimulationParameters.viscosityStrength.value)
                * 2.0f;
            object->tempValue2.as_float2.x = density;
        }
        warp.sync();
        if (profiling) {
            auto const now = readSmClock();
            profileTailCycles += now - profileMark;
            if (profileObjects == 1) {
                profileFirstObjectCycles = now - profileObjectStartCycles;
            }
            profileLoopCycles += now - profileObjectStartCycles;
            profileMark = now;
        }
    }

    if (profiling) {
        auto const blockEndNs = readGlobalTimer();
        auto const warpEndCycles = readSmClock();
        atomicAdd(&profileScanCells, threadScanCells);
        atomicAdd(&profileRecords, threadRecords);
        auto& profile = cudaFluidKernelProfile;
        if (warp.thread_rank() == 0) {
            atomicAdd(&profile.initCycles, profileInitCycles);
            atomicAdd(&profile.scanCycles, profileScanCycles);
            atomicAdd(&profile.syncAndReduceCycles, profileSyncCycles);
            atomicAdd(&profile.tailCycles, profileTailCycles);
            atomicAdd(&profile.warpCycles, warpEndCycles - profileWarpStartCycles);
            atomicAdd(&profile.warpOverheadCycles, warpEndCycles - profileWarpStartCycles - profileLoopCycles);
            atomicAdd(&profile.numWarps, 1);
            atomicAdd(&profile.numObjects, profileObjects);
            atomicAdd(&profile.firstObjectCycles, profileFirstObjectCycles);
        }
        block.sync();
        if (block.thread_rank() == 0) {
            atomicAdd(&profile.blockNanoseconds, blockEndNs - profileBlockStartNs);
            atomicAdd(&profile.numBlocks, 1);
            atomicAdd(&profile.numScanCells, profileScanCells);
            atomicAdd(&profile.numRecords, profileRecords);
            if (blockIdx.x == 0) {
                atomicAdd(&profile.numLaunches, 1);
            }
        }
    }
}

__inline__ __device__ void ObjectProcessor::calcFluidBoundaryForces(SimulationData& data)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    auto& objects = data.entities.objects;
    auto const warpPartition = calcWarpPartition(objects.getNumEntries());
    auto const smoothingLength = cudaSimulationParameters.smoothingLength.value * 2.0f;  // Fluid uses 2x base smoothing length

    for (int objectIndex = warpPartition.startIndex; objectIndex <= warpPartition.endIndex; ++objectIndex) {
        auto& object = objects.at(objectIndex);

        if (object->type != ObjectType_Fluid) {
            continue;
        }

        int const radiusInt = ceilf(smoothingLength * 2);
        int const scanLength = radiusInt * 2 + 1;
        int2 const cellPosInt = {floorInt(object->pos.x) - radiusInt, floorInt(object->pos.y) - radiusInt};

        float2 localF_boundary = {0, 0};

        auto const objectPos = object->pos;
        auto const cutoffSquared = smoothingLength * smoothingLength * 4;

        auto const invScanLength = 1.0f / toFloat(scanLength);

        auto records = data.objectMap.getRecords();
        for (int scanIndex = toInt(warp.thread_rank()); scanIndex < scanLength * scanLength; scanIndex += warp.size()) {
            int2 scanPos = calcScanPos(cellPosInt, scanIndex, scanLength, invScanLength);

            if (!isCellInRange(objectPos, scanPos.x, scanPos.y, cutoffSquared)) {
                continue;
            }
            data.objectMap.correctPosition(scanPos);
            int otherIndex = data.objectMap.getFirstIndex(scanPos);
            for (int level = 0; level < MaxBarrierCellsForCollision; ++level) {
                if (otherIndex < 0) {
                    break;
                }
                auto const& other = &records[otherIndex];
                auto otherObject = other->self;  // Read fields live: this runs after calcFluidForces nudged positions
                if (other->type != ObjectType_Fluid && otherObject != object && object->detached() + otherObject->detached() != 1) {

                    auto posDelta = object->pos - otherObject->pos;
                    data.objectMap.correctDirection(posDelta);
                    auto adaptedDistance = Math::length(posDelta);

                    if (adaptedDistance <= smoothingLength * 2 && adaptedDistance > NEAR_ZERO) {
                        auto solidMass = getMassForSPH(other);

                        float kernel_d_val = calcKernel_d(adaptedDistance / smoothingLength) / (smoothingLength * smoothingLength * smoothingLength);

                        // Repulsion force on fluid from solid boundary.
                        // Factor 2/rho_f mirrors the symmetric SPH pressure factor (1/rho_f + 1/rho_f)
                        // and is proportional to solid mass so that a heavier boundary repels more strongly.
                        auto F_on_fluid = posDelta / (-adaptedDistance) * (2.0f / max(NEAR_ZERO, object->density)) * kernel_d_val * solidMass * 0.3f;
                        localF_boundary += F_on_fluid;

                        // Counter-force on solid: equal and opposite (Newton's 3rd law).
                        // pressureStrength is applied here directly since this force bypasses the
                        // warp-local F_boundary accumulation path (which gets pressureStrength at the end).
                        atomicAdd(&otherObject->tempValue1.as_float2.x, -F_on_fluid.x * cudaSimulationParameters.pressureStrength.value);
                        atomicAdd(&otherObject->tempValue1.as_float2.y, -F_on_fluid.y * cudaSimulationParameters.pressureStrength.value);
                    }
                }
                otherIndex = other->nextObjectIndex;
            }
        }

        // The warp owns the object, so its reduction already yields the total.
        float2 const F_boundary{cg::reduce(warp, localF_boundary.x, cg::plus<float>()), cg::reduce(warp, localF_boundary.y, cg::plus<float>())};

        if (warp.thread_rank() == 0) {
            object->tempValue1.as_float2 += F_boundary * cudaSimulationParameters.pressureStrength.value;
        }
        warp.sync();
    }
}

__inline__ __device__ void ObjectProcessor::checkForces(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        object->density = object->tempValue2.as_float2.x;
        if (object->isStatic()) {
            continue;
        }

        if (Math::length(object->tempValue1.as_float2)
            > ParameterCalculator::calcParameter(cudaSimulationParameters.maxForce, data, object->pos, object->color)) {
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
        if (object->isStatic()) {
            continue;
        }
        auto acceleration = object->tempValue1.as_float2 / max(0.05f, object->density) * 0.5f;
        if (Math::length(acceleration) > cudaSimulationParameters.maxAcceleration) {
            acceleration = Math::getNormalized(acceleration) * cudaSimulationParameters.maxAcceleration;
        }
        object->vel += acceleration;
        if (Math::length(object->vel) > cudaSimulationParameters.maxVelocity.value) {
            object->vel = Math::getNormalized(object->vel) * cudaSimulationParameters.maxVelocity.value;
        }
        object->tempValue1.as_float2 = {0, 0};
    }
}

__inline__ __device__ void ObjectProcessor::calcConnectionForces(SimulationData& data, bool calcAngularForces)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (0 == object->numConnections /* || object->isStatic()*/) {
            continue;
        }
        float2 force{0, 0};
        float2 prevDisplacement = object->connections[object->numConnections - 1].object->pos - object->pos;
        data.objectMap.correctDirection(prevDisplacement);
        auto cellStiffnessSquared = object->stiffness * object->stiffness;

        auto numConnections = object->numConnections;
        auto prevAngle = calcAngularForces ? Math::angleOfVector(prevDisplacement) : 0.0f;
        for (int i = 0; i < numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            auto connectedObjectStiffnessSquared = connectedObject->stiffness * connectedObject->stiffness;

            auto displacement = connectedObject->pos - object->pos;
            data.objectMap.correctDirection(displacement);

            auto actualDistance = Math::length(displacement);
            auto bondDistance = object->connections[i].distance;
            auto deviation = actualDistance - bondDistance;
            auto direction = actualDistance > NEAR_ZERO ? displacement / actualDistance : float2{1.0f, 0.0f};
            force = force + direction * deviation * (cellStiffnessSquared + connectedObjectStiffnessSquared) / 6;
            if (calcAngularForces) {
                auto lastIndex = (i + numConnections - 1) % numConnections;
                auto lastConnectedObject = object->connections[lastIndex].object;

                auto referenceAngleFromPrevious = object->connections[i].angleFromPrevious;

                auto r1 = prevDisplacement;
                auto r2 = displacement;
                Math::rotateQuarterClockwise(r1);
                Math::rotateQuarterCounterClockwise(r2);

                auto angle = Math::angleOfVector(displacement);
                auto theta = Math::getNormalizedAngle(angle - prevAngle, 0.0f);
                prevAngle = angle;

                if (theta < referenceAngleFromPrevious) {
                    r1 *= -1.0f;
                    r2 *= -1.0f;
                }
                auto g = 5e-4f * abs(Math::getNormalizedAngle(theta - referenceAngleFromPrevious, -180.0f)) * cellStiffnessSquared;
                auto strength1 = g / max(Math::lengthSquared(r1), 0.1f);
                auto strength2 = g / max(Math::lengthSquared(r2), 0.1f);
                auto force2 = r1 * strength1;
                auto force1 = r2 * strength2;

                atomicAdd(&connectedObject->tempValue1.as_float2.x, force1.x);
                atomicAdd(&connectedObject->tempValue1.as_float2.y, force1.y);
                atomicAdd(&lastConnectedObject->tempValue1.as_float2.x, force2.x);
                atomicAdd(&lastConnectedObject->tempValue1.as_float2.y, force2.y);
                force -= force1 + force2;
            }

            prevDisplacement = displacement;
        }
        atomicAdd(&object->tempValue1.as_float2.x, force.x);
        atomicAdd(&object->tempValue1.as_float2.y, force.y);
    }
}

__inline__ __device__ void ObjectProcessor::checkConnections(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->isStatic()) {
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
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletPositionUpdate(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->isStatic()) {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value;
            data.objectMap.correctPosition(object->pos);
        } else {
            object->pos += object->vel * cudaSimulationParameters.timestepSize.value
                + object->tempValue1.as_float2 * cudaSimulationParameters.timestepSize.value * cudaSimulationParameters.timestepSize.value / 2;
            data.objectMap.correctPosition(object->pos);
            object->tempValue2.as_float2 = object->tempValue1.as_float2;  // Save forces from first step for averaging
            object->tempValue1.as_float2 = {0, 0};
        }
    }
}

__inline__ __device__ void ObjectProcessor::verletVelocityUpdate(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto const partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->isStatic()) {
            continue;
        }
        auto acceleration = (object->tempValue1.as_float2 + object->tempValue2.as_float2) / 2;
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
        if (object->isStatic()) {
            continue;
        }
        for (int index = 0; index < object->numConnections; ++index) {
            auto connectedObject = object->connections[index].object;
            if (connectedObject->isStatic()) {
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
        if (object->isStatic()) {
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
        if (object->isStatic()) {
            continue;
        }
        if (object->type == ObjectType_Solid || object->type == ObjectType_Fluid) {
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
                    - particleVel;  // Minus particleVel because particle will still be moved in current time step
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
