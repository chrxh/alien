#pragma once

#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "ParameterCalculator.cuh"
#include "TO.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void collectCellTypeOperations(SimulationData& data);
    __inline__ __device__ static void calcFutureSignals(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static void createEmptySignal(Cell* cell);
    __inline__ __device__ static float2 calcReferenceDirection(SimulationData& data, Cell* cell);

    __inline__ __device__ static bool isAutoTriggered(SimulationData& data, Cell* cell, uint32_t autoTriggerInterval, bool isPreview = false);
    __inline__ __device__ static bool isManuallyTriggered(SimulationData& data, Cell* cell);
    __inline__ __device__ static bool isAutoOrManuallyTriggered(SimulationData& data, Cell* cell, uint32_t autoTriggerInterval, bool isPreview = false);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SignalProcessor::collectCellTypeOperations(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);

        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free && cell->cellType != CellType_Base) {
            if (cell->cellType == CellType_Detonator && cell->cellTypeData.detonator.state == DetonatorState_Activated) {
                data.cellTypeOperations[cell->cellType].tryAddEntry(CellTypeOperation{cell});
            } else if (cell->cellState != CellState_Constructing && cell->cellState != CellState_Activating && cell->activationTime == 0) {
                data.cellTypeOperations[cell->cellType].tryAddEntry(CellTypeOperation{cell});
            }
        }
    }
}

__inline__ __device__ void SignalProcessor::calcFutureSignals(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->cellType == CellType_Structure || cell->cellType == CellType_Free) {
            continue;
        }

        cell->futureSignalState = SignalState_Inactive;
        if (cell->signalState != SignalState_Inactive) {
            continue;
        }

        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = 0;
        }
        cell->futureSignal.numTimesSent = INT_MAX;  // Will track minimum

        for (int i = 0, j = cell->numConnections; i < j; ++i) {
            auto connectedCell = cell->connections[i].cell;
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
                    if (connectedCell->connections[k].cell == cell) {
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

            cell->futureSignalState = SignalState_Active;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                cell->futureSignal.channels[k] += connectedCell->signal.channels[k];
            }
            // Keep minimum numTimesSent when signals merge
            cell->futureSignal.numTimesSent = min(cell->futureSignal.numTimesSent, connectedCell->signal.numTimesSent);
        }

        // Reset numTimesSent to 0 if no signals were received
        if (cell->futureSignal.numTimesSent == INT_MAX) {
            cell->futureSignal.numTimesSent = 0;
        }
    }
}

__inline__ __device__ void SignalProcessor::updateSignals(SimulationData& data)
{
    auto& cells = data.objects.cells;
    auto partition = calcSystemThreadPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& cell = cells.at(index);
        if (cell->cellType == CellType_Structure || cell->cellType == CellType_Free) {
            continue;
        }

        if (cell->futureSignalState != SignalState_Inactive) {
            cell->cellTriggered = CellTriggered_Yes;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->signal.channels[i] = cell->futureSignal.channels[i];
            }
            cell->signal.numTimesSent = cell->futureSignal.numTimesSent;
            cell->signalState = SignalState_Active;
        } else {
            cell->signalState = max(0, cell->signalState - 1);
        }
    }
}

__inline__ __device__ void SignalProcessor::createEmptySignal(Cell* cell)
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = 0;
    }
    cell->signal.numTimesSent = 0;
    cell->signalState = SignalState_Active;
}

__inline__ __device__ float2 SignalProcessor::calcReferenceDirection(SimulationData& data, Cell* cell)
{
    if (cell->numConnections == 0) {
        return float2{0.0f, -1.0f};
    }
    return Math::getNormalized(data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos));
}

__inline__ __device__ bool SignalProcessor::isAutoTriggered(SimulationData& data, Cell* cell, uint32_t autoTriggerInterval, bool isPreview)
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

__inline__ __device__ bool SignalProcessor::isManuallyTriggered(SimulationData& data, Cell* cell)
{
    if (cell->signalState != SignalState_Active) {
        return false;
    }
    if (abs(cell->signal.channels[Channels::CellTypeActivation]) < TRIGGER_THRESHOLD) {
        return false;
    }
    return true;
}

__inline__ __device__ bool SignalProcessor::isAutoOrManuallyTriggered(SimulationData& data, Cell* cell, uint32_t autoTriggerInterval, bool isPreview)
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
