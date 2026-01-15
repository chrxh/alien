#pragma once

#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "TO.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void collectCellTypeOperations(SimulationData& data);
    __inline__ __device__ static void calcFutureSignals(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static void createEmptySignal(Object* cell);
    __inline__ __device__ static float2 calcReferenceDirection(SimulationData& data, Object* cell);

    __inline__ __device__ static bool isAutoTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview = false);
    __inline__ __device__ static bool isManuallyTriggered(SimulationData& data, Object* cell);
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

        if (object->cellType != CellType_Structure && object->cellType != CellType_Free && object->cellType != CellType_Base) {
            if (object->cellType == CellType_Detonator && object->cellTypeData.detonator.state == DetonatorState_Activated) {
                data.cellTypeOperations[object->cellType].tryAddEntry(CellTypeOperation{object});
            } else if (object->cellState != CellState_Constructing && object->cellState != CellState_Activating && object->activationTime == 0) {
                data.cellTypeOperations[object->cellType].tryAddEntry(CellTypeOperation{object});
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
        if (object->cellType == CellType_Structure || object->cellType == CellType_Free) {
            continue;
        }

        object->futureSignalState = SignalState_Inactive;
        if (object->signalState != SignalState_Inactive) {
            continue;
        }

        for (int i = 0; i < MAX_CHANNELS; ++i) {
            object->futureSignal.channels[i] = 0;
        }
        object->futureSignal.numTimesSent = INT_MAX;  // Will track minimum

        for (int i = 0, j = object->numConnections; i < j; ++i) {
            auto connectedCell = object->connections[i].object;
            if (connectedCell->cellState == CellState_Constructing || connectedCell->signalState != SignalState_Active) {
                continue;
            }
            int skip = false;
            auto restrictionMode = connectedCell->signalRestriction.mode;
            
            if (restrictionMode == SignalRestrictionMode_Active || restrictionMode == SignalRestrictionMode_Conditional) {
                float signalAngleRestrictionStart = 180.0f + connectedCell->signalRestriction.baseAngle - connectedCell->signalRestriction.openingAngle / 2;
                float signalAngleRestrictionEnd = 180.0f + connectedCell->signalRestriction.baseAngle + connectedCell->signalRestriction.openingAngle / 2;

                float connectionAngle = 0;
                for (int k = 0, l = connectedCell->numConnections; k < l; ++k) {
                    if (connectedCell->connections[k].object == object) {
                        bool isInsideCone = Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, connectionAngle);
                        if (!isInsideCone) {
                            // Outside the cone: signal is always blocked for both Active and Conditional modes
                            skip = true;
                        } else if (restrictionMode == SignalRestrictionMode_Conditional) {
                            // Inside the cone in Conditional mode: signal passes only if channel[0] >= 0
                            if (connectedCell->signal.channels[0] < 0) {
                                skip = true;
                            }
                        }
                        break;
                    }
                    connectionAngle += connectedCell->connections[k].angleFromPrevious;
                }
                if (skip) {
                    continue;
                }
            }

            object->futureSignalState = SignalState_Active;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                object->futureSignal.channels[k] += connectedCell->signal.channels[k];
            }
            // Keep minimum numTimesSent when signals merge
            object->futureSignal.numTimesSent = min(object->futureSignal.numTimesSent, connectedCell->signal.numTimesSent);
        }
    }
}

__inline__ __device__ void SignalProcessor::updateSignals(SimulationData& data)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (object->cellType == CellType_Structure || object->cellType == CellType_Free) {
            continue;
        }

        if (object->futureSignalState != SignalState_Inactive) {
            object->cellTriggered = CellTriggered_Yes;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                object->signal.channels[i] = object->futureSignal.channels[i];
            }
            object->signal.numTimesSent = object->futureSignal.numTimesSent;
            object->signalState = SignalState_Active;
        } else {
            object->signalState = max(0, object->signalState - 1);
        }
    }
}

__inline__ __device__ void SignalProcessor::createEmptySignal(Object* cell)
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = 0;
    }
    cell->signal.numTimesSent = 0;
    cell->signalState = SignalState_Active;
}

__inline__ __device__ float2 SignalProcessor::calcReferenceDirection(SimulationData& data, Object* cell)
{
    if (cell->numConnections == 0) {
        return float2{0.0f, -1.0f};
    }
    return Math::getNormalized(data.cellMap.getCorrectedDirection(cell->connections[0].object->pos - cell->pos));
}

__inline__ __device__ bool SignalProcessor::isAutoTriggered(SimulationData& data, Object* cell, uint32_t autoTriggerInterval, bool isPreview)
{
    auto triggerInterval = max(SignalState_Count, autoTriggerInterval);
    if (cell->creature != nullptr) {
        if (isPreview) {
            return *data.timestep % triggerInterval == 0;
        } else {
            return (*data.timestep + cell->creature->id) % triggerInterval == 0;
        }
    } else {
        return *data.timestep % triggerInterval == 0;
    }
}

__inline__ __device__ bool SignalProcessor::isManuallyTriggered(SimulationData& data, Object* cell)
{
    if (cell->signalState != SignalState_Active) {
        return false;
    }
    if (abs(cell->signal.channels[Channels::CellTypeActivation]) < TRIGGER_THRESHOLD) {
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
