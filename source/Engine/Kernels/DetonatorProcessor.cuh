#pragma once

#include <Data/Interface/CellTypeConstants.h>

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

    enum class Action
    {
        None,
        Detonate,
        PropagateShockWave
    };

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static Action updateState(SimulationData& data, Object* object);
    __inline__ __device__ static void detonate(SimulationData& data, Object* object);
    __inline__ __device__ static void propagateShockWave(SimulationData& data, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DetonatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Detonator];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ Action action;
    if (threadIdx.x == 0) {
        action = updateState(data, object);
    }
    __syncthreads();

    if (action == Action::Detonate) {
        detonate(data, object);
    } else if (action == Action::PropagateShockWave) {
        propagateShockWave(data, object);
    }
    __syncthreads();
}

__device__ __inline__ DetonatorProcessor::Action DetonatorProcessor::updateState(SimulationData& data, Object* object)
{
    auto& detonator = object->typeData.cell.cellTypeData.detonator;
    if (detonator.state == DetonatorState_Exploded) {
        return Action::PropagateShockWave;
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
            detonator.state = DetonatorState_Exploded;
            return Action::Detonate;
        }
    }
    return Action::None;
}

__device__ __inline__ void DetonatorProcessor::detonate(SimulationData& data, Object* object)
{
    auto radius = cudaSimulationParameters.detonatorRadius.value[object->color];
    auto chainExplosionProbability = cudaSimulationParameters.detonatorChainExplosionProbability.value[object->color];
    data.objectGrid.executeForEach_block(object->pos, radius, object->detached(), [&](Object* const& otherObject) {
        if (otherObject == object) {
            return;
        }
        if (otherObject->isStatic()) {
            return;
        }
        auto delta = data.world.getCorrectedDirection(otherObject->pos - object->pos);
        auto lengthSquared = Math::lengthSquared(delta);
        if (lengthSquared > NEAR_ZERO) {
            auto force = delta / lengthSquared * radius * 2;
            otherObject->vel += force;
        }
        if (otherObject->typeData.cell.cellType == CellType_Detonator && otherObject->typeData.cell.cellTypeData.detonator.state != DetonatorState_Exploded) {
            if (data.primaryNumberGen.random() < chainExplosionProbability) {
                otherObject->typeData.cell.cellTypeData.detonator.state = DetonatorState_Activated;
                otherObject->typeData.cell.cellTypeData.detonator.countdown = 1;
            }
        }
    });
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
    data.objectGrid.executeForEachInRing_block(
        object->pos, calcFrontRadius(step - 1), calcFrontRadius(step), object->detached(), [&](Object* const& otherObject) {
            if (otherObject->isStatic()) {
                return;
            }
            auto delta = data.world.getCorrectedDirection(otherObject->pos - object->pos);
            auto distance = Math::length(delta);
            auto direction = delta / distance;
            auto flowVelocity = ShockWaveStrength * sqrtf(radius / distance);
            auto radialVelocity = Math::dot(otherObject->vel, direction);
            if (radialVelocity < flowVelocity) {
                otherObject->vel += direction * (flowVelocity - radialVelocity);
            }
        });
}
