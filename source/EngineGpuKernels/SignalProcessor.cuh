#pragma once

#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "TOs.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void collectCellTypeOperations(SimulationData& data);
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

__inline__ __device__ void SignalProcessor::collectCellTypeOperations(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);

        if (object->type == ObjectType_Cell && object->typeData.cell.cellType != CellType_Base) {
            if (object->typeData.cell.cellType == CellType_Detonator && object->typeData.cell.cellTypeData.detonator.state == DetonatorState_Activated) {
                data.cellTypeOperations[object->typeData.cell.cellType].tryAddEntry(CellTypeOperation{object});
            } else if (object->typeData.cell.cellState != CellState_Constructing && object->typeData.cell.cellState != CellState_Activating && object->typeData.cell.activationTime == 0) {
                data.cellTypeOperations[object->typeData.cell.cellType].tryAddEntry(CellTypeOperation{object});
            }
        }
        // Also collect cells with constructorAvailable for constructor operations
        if (object->type == ObjectType_Cell && object->typeData.cell.constructorAvailable) {
            if (object->typeData.cell.cellState != CellState_Constructing && object->typeData.cell.cellState != CellState_Activating && object->typeData.cell.activationTime == 0) {
                data.constructorOperations.tryAddEntry(CellTypeOperation{object});
            }
        }
    }
}

__inline__ __device__ void SignalProcessor::calcFutureSignals(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type == ObjectType_Structure || object->type == ObjectType_FreeCell) {
            continue;
        }

        object->typeData.cell.futureSignalState = SignalState_Inactive;
        if (object->typeData.cell.signalState != SignalState_Inactive) {
            continue;
        }

        for (int i = 0; i < MAX_CHANNELS; ++i) {
            object->typeData.cell.futureSignal.channels[i] = 0;
        }
        object->typeData.cell.futureSignal.numTimesSent = INT_MAX;  // Will track minimum

        for (int i = 0, j = object->numConnections; i < j; ++i) {
            auto connectedObject = object->connections[i].object;
            // Skip Structure and FreeCell objects for signal propagation
            if (connectedObject->type == ObjectType_Structure || connectedObject->type == ObjectType_FreeCell) {
                continue;
            }
            if (connectedObject->typeData.cell.cellState == CellState_Constructing || connectedObject->typeData.cell.signalState != SignalState_Active) {
                continue;
            }
            int skip = false;
            auto restrictionMode = connectedObject->typeData.cell.signalRestriction.mode;
            
            if (restrictionMode == SignalRestrictionMode_Active || restrictionMode == SignalRestrictionMode_Conditional) {
                float signalAngleRestrictionStart = 180.0f + connectedObject->typeData.cell.signalRestriction.baseAngle - connectedObject->typeData.cell.signalRestriction.openingAngle / 2;
                float signalAngleRestrictionEnd = 180.0f + connectedObject->typeData.cell.signalRestriction.baseAngle + connectedObject->typeData.cell.signalRestriction.openingAngle / 2;

                float connectionAngle = 0;
                for (int k = 0, l = connectedObject->numConnections; k < l; ++k) {
                    if (connectedObject->connections[k].object == object) {
                        bool isInsideCone = Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, connectionAngle);
                        if (!isInsideCone) {
                            // Outside the cone: signal is always blocked for both Active and Conditional modes
                            skip = true;
                        } else if (restrictionMode == SignalRestrictionMode_Conditional) {
                            // Inside the cone in Conditional mode: signal passes only if channel[0] >= 0
                            if (connectedObject->typeData.cell.signal.channels[0] < 0) {
                                skip = true;
                            }
                        }
                        break;
                    }
                    connectionAngle += connectedObject->connections[k].angleFromPrevious;
                }
                if (skip) {
                    continue;
                }
            }

            object->typeData.cell.futureSignalState = SignalState_Active;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                object->typeData.cell.futureSignal.channels[k] += connectedObject->typeData.cell.signal.channels[k];
            }
            // Keep minimum numTimesSent when signals merge
            object->typeData.cell.futureSignal.numTimesSent = min(object->typeData.cell.futureSignal.numTimesSent, connectedObject->typeData.cell.signal.numTimesSent);
        }
    }
}

__inline__ __device__ void SignalProcessor::updateSignals(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->type == ObjectType_Structure || object->type == ObjectType_FreeCell) {
            continue;
        }

        if (object->typeData.cell.futureSignalState != SignalState_Inactive) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                object->typeData.cell.signal.channels[i] = object->typeData.cell.futureSignal.channels[i];
            }
            object->typeData.cell.signal.numTimesSent = object->typeData.cell.futureSignal.numTimesSent;
            object->typeData.cell.signalState = SignalState_Active;
        } else {
            object->typeData.cell.signalState = max(0, object->typeData.cell.signalState - 1);
        }
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
