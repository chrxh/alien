#pragma once

#include <Data/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DetonatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    static auto constexpr DetonationEventDuration = 30;  // In cell function cycles
    static auto constexpr ShockWaveDuration = 8;         // In cell function cycles
    static auto constexpr ShockWaveReach = 8.0f;         // In detonator radii
    static auto constexpr ShockWaveStrength = 1.2f;

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void propagateShockWave(SimulationData& data, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DetonatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Detonator];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& detonator = object->typeData.cell.cellTypeData.detonator;
    if (detonator.state == DetonatorState_Exploded) {
        propagateShockWave(data, object);
        return;
    }
    if (NeuronProcessor::isManuallyTriggered(data, object) && detonator.state == DetonatorState_Ready) {
        detonator.state = DetonatorState_Activated;
    }
    if (detonator.state == DetonatorState_Activated) {
        if (detonator.countdown >= 0) {
            --detonator.countdown;
        }
        if (detonator.countdown == -1) {
            object->typeData.cell.event = CellEvent_Detonation;
            object->typeData.cell.eventCounter = DetonationEventDuration;
            detonator.countdown = 0;
            data.objectMap.executeForEach(
                object->pos, cudaSimulationParameters.detonatorRadius.value[object->color], object->detached(), [&](Object* const& otherObject) {
                    if (otherObject == object) {
                        return;
                    }
                    if (otherObject->isStatic()) {
                        return;
                    }
                    auto delta = data.objectMap.getCorrectedDirection(otherObject->pos - object->pos);
                    auto lengthSquared = Math::lengthSquared(delta);
                    if (lengthSquared > NEAR_ZERO) {
                        auto force = delta / lengthSquared * cudaSimulationParameters.detonatorRadius.value[object->color] * 2;
                        otherObject->vel += force;
                    }
                    if (otherObject->typeData.cell.cellType == CellType_Detonator
                        && otherObject->typeData.cell.cellTypeData.detonator.state != DetonatorState_Exploded) {
                        if (data.primaryNumberGen.random() < cudaSimulationParameters.detonatorChainExplosionProbability.value[object->color]) {
                            otherObject->typeData.cell.cellTypeData.detonator.state = DetonatorState_Activated;
                            otherObject->typeData.cell.cellTypeData.detonator.countdown = 1;
                        }
                    }
                });
            detonator.state = DetonatorState_Exploded;
        }
    }
}

__device__ __inline__ void DetonatorProcessor::propagateShockWave(SimulationData& data, Object* object)
{
    auto const& cell = object->typeData.cell;
    auto step = DetonationEventDuration - cell.eventCounter;
    if (cell.event != CellEvent_Detonation || step < 1 || step > ShockWaveDuration) {
        return;
    }
    auto radius = cudaSimulationParameters.detonatorRadius.value[object->color];
    auto calcFrontRadius = [&](int step) { return radius * (1.0f + (ShockWaveReach - 1.0f) * toFloat(step) / toFloat(ShockWaveDuration)); };

    // The front sweeps a new ring in each cell function cycle and accelerates matter up to the flow velocity behind it
    data.objectMap.executeForEachInRing(object->pos, calcFrontRadius(step - 1), calcFrontRadius(step), object->detached(), [&](Object* const& otherObject) {
        if (otherObject->isStatic()) {
            return;
        }
        auto delta = data.objectMap.getCorrectedDirection(otherObject->pos - object->pos);
        auto distance = Math::length(delta);
        auto direction = delta / distance;
        auto flowVelocity = ShockWaveStrength * sqrtf(radius / distance);
        auto radialVelocity = Math::dot(otherObject->vel, direction);
        if (radialVelocity < flowVelocity) {
            otherObject->vel += direction * (flowVelocity - radialVelocity);
        }
    });
}
