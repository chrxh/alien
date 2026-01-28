#pragma once

#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "TOs.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void calcFutureSignals(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static void createEmptySignal(Object* object);
    __inline__ __device__ static float2 calcReferenceDirection(SimulationData& data, Object* object);

    __inline__ __device__ static bool isAutoTriggered(SimulationData& data, Object* object, uint32_t autoTriggerInterval, bool isPreview = false);
    __inline__ __device__ static bool isManuallyTriggered(SimulationData& data, Object* object);
    __inline__ __device__ static bool isAutoOrManuallyTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview = false);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SignalProcessor::calcFutureSignals(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& cell = object->typeData.cell;
        if (cell.cellState == CellState_Constructing) {
            continue;
        }

        clearChannels(object->typeData.cell.futureSignal.channels);
        object->typeData.cell.futureSignal.numTimesSent = INT_MAX;  // Will track minimum

        for (int i = 0, j = object->numConnections; i < j; ++i) {
            auto const& connectedObject = object->connections[i].object;
         
            if (connectedObject->type != ObjectType_Cell) {
                continue;
            }
            auto& connectedCell = connectedObject->typeData.cell;
            if (connectedCell.cellState == CellState_Constructing) {
                continue;
            }

            addChannelsWithWeight(cell.futureSignal.channels, connectedCell.signal.channels, cell.neuralNetwork->connectionWeights[i]);
            cell.futureSignal.numTimesSent = min(cell.futureSignal.numTimesSent, connectedCell.signal.numTimesSent);
        }
    }
}

__inline__ __device__ void SignalProcessor::updateSignals(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& cell = object->typeData.cell;

        addChannels(cell.signal.channels, cell.futureSignal.channels);
        cell.signal.numTimesSent = object->typeData.cell.futureSignal.numTimesSent;
    }
}

__inline__ __device__ void SignalProcessor::createEmptySignal(Object* object)
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        object->typeData.cell.signal.channels[i] = 0;
    }
    object->typeData.cell.signal.numTimesSent = 0;
    object->typeData.cell.signalState = SignalState_Active;
}

__inline__ __device__ float2 SignalProcessor::calcReferenceDirection(SimulationData& data, Object* object)
{
    if (object->numConnections == 0) {
        return float2{0.0f, -1.0f};
    }
    return Math::getNormalized(data.objectMap.getCorrectedDirection(object->connections[0].object->pos - object->pos));
}

__inline__ __device__ bool SignalProcessor::isAutoTriggered(SimulationData& data, Object* object, uint32_t autoTriggerInterval, bool isPreview)
{
    CUDA_CHECK(object->type == ObjectType_Cell);

    auto triggerInterval = max(SignalState_Count, autoTriggerInterval);
    if (isPreview) {
        return *data.timestep % triggerInterval == 0;
    } else {
        return (*data.timestep + object->typeData.cell.creature->id) % triggerInterval == 0;
    }
}

__inline__ __device__ bool SignalProcessor::isManuallyTriggered(SimulationData& data, Object* object)
{
    if (object->typeData.cell.signalState != SignalState_Active) {
        return false;
    }
    if (abs(object->typeData.cell.signal.channels[Channels::CellTypeActivation]) < TRIGGER_THRESHOLD) {
        return false;
    }
    return true;
}

__inline__ __device__ bool SignalProcessor::isAutoOrManuallyTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview)
{
    if (autoTriggerInterval == 0) {
        return isManuallyTriggered(data, cell);
    } else {
        if (!isAutoTriggered(data, cell, autoTriggerInterval, isPreview)) {
            return false;
        }
    }
    return true;
}
