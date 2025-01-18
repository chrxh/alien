#pragma once

#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "SpotCalculator.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void collectCellTypeOperations(SimulationData& data);
    __inline__ __device__ static void calcFutureSignals(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static void createEmptySignal(Cell* cell);
    __inline__ __device__ static float2 calcSignalDirection(SimulationData& data, Cell* cell);
    __inline__ __device__ static bool isTriggeredAndCreateSignalIfTriggered(SimulationData& data, Cell* cell, uint8_t autoTriggerInterval);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SignalProcessor::collectCellTypeOperations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->cellType != CellType_Structure && cell->cellType != CellType_Free && cell->cellType != CellType_Base) {
            if (cell->cellType == CellType_Detonator && cell->cellTypeData.detonator.state == DetonatorState_Activated) {
                data.cellTypeOperations[cell->cellType].tryAddEntry(CellTypeOperation{cell});
            } else if (cell->livingState != LivingState_UnderConstruction && cell->livingState != LivingState_Activating && cell->activationTime == 0) {
                data.cellTypeOperations[cell->cellType].tryAddEntry(CellTypeOperation{cell});
            }

        }
    }
}

__inline__ __device__  void SignalProcessor::calcFutureSignals(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        cell->futureSignal.active = false;
        if (cell->signalRelaxationTime > 0) {
            continue;
        }
        cell->futureSignal.origin = SignalOrigin_Unknown;
        cell->futureSignal.targetX = 0;
        cell->futureSignal.targetY = 0;

        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = 0;
        }

        int numSensorSignals = 0;
        for (int i = 0, j = cell->numConnections; i < j; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->livingState == LivingState_UnderConstruction || !connectedCell->signal.active) {
                continue;
            }
            int skip = false;
            if (connectedCell->signalRoutingRestriction.active) {
                float signalAngleRestrictionStart = 0;
                float signalAngleRestrictionEnd = 0;
                if (connectedCell->signalRoutingRestriction.active) {
                    signalAngleRestrictionStart =
                        180.0f + connectedCell->signalRoutingRestriction.baseAngle - connectedCell->signalRoutingRestriction.openingAngle / 2;
                    signalAngleRestrictionEnd =
                        180.0f + connectedCell->signalRoutingRestriction.baseAngle + connectedCell->signalRoutingRestriction.openingAngle / 2;
                }

                float connectionAngle = 0;
                for (int k = 0, l = connectedCell->numConnections; k < l; ++k) {
                    if (connectedCell->connections[k].cell == cell) {
                        if (connectionAngle < signalAngleRestrictionStart - NEAR_ZERO || connectionAngle > signalAngleRestrictionEnd + NEAR_ZERO) {
                            skip = true;
                        }
                        break;
                    }
                    connectionAngle += connectedCell->connections[k].angleFromPrevious;
                }
                if (skip) {
                    continue;
                }
            }

            cell->futureSignal.active = true;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                cell->futureSignal.channels[k] += connectedCell->signal.channels[k];
            }
            if (connectedCell->signal.origin == SignalOrigin_Sensor) {
                cell->futureSignal.origin = SignalOrigin_Sensor;
                cell->futureSignal.targetX += connectedCell->signal.targetX;
                cell->futureSignal.targetY += connectedCell->signal.targetY;
                ++numSensorSignals;
            }
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = max(-10.0f, min(10.0f, cell->futureSignal.channels[i]));  // truncate value to avoid overflow
        }
        if (numSensorSignals > 0) {
            cell->futureSignal.targetX /= toFloat(numSensorSignals);
            cell->futureSignal.targetY /= toFloat(numSensorSignals);
        }
    }
}

__inline__ __device__ void SignalProcessor::updateSignals(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        cell->signal.active = cell->futureSignal.active;
        if (cell->signal.active) {
            cell->cellTypeUsed = CellTriggered_Yes;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->signal.channels[i] = cell->futureSignal.channels[i];
            }
            cell->signal.origin = cell->futureSignal.origin;
            cell->signal.targetX = cell->futureSignal.targetX;
            cell->signal.targetY = cell->futureSignal.targetY;
            cell->signalRelaxationTime = MAX_SIGNAL_RELAXATION_TIME;
        } else {
            cell->signalRelaxationTime = max(0, cell->signalRelaxationTime - 1);
        }
    }
}

__inline__ __device__ void SignalProcessor::createEmptySignal(Cell* cell)
{
    cell->signal.active = true;
    cell->signal.origin = SignalOrigin_Unknown;
    cell->signal.targetX = 0;
    cell->signal.targetY = 0;
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->signal.channels[i] = 0;
    }
    cell->signalRelaxationTime = MAX_SIGNAL_RELAXATION_TIME;
}

__inline__ __device__ float2 SignalProcessor::calcSignalDirection(SimulationData& data, Cell* cell)
{
    float2 result{0, 0};
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectedCell = cell->connections[i].cell;
        auto directionDelta = cell->pos - connectedCell->pos;
        data.cellMap.correctDirection(directionDelta);
        result = result + Math::normalized(directionDelta);
    }
    return Math::normalized(result);
}

__inline__ __device__ bool SignalProcessor::isTriggeredAndCreateSignalIfTriggered(SimulationData& data, Cell* cell, uint8_t autoTriggerInterval)
{
    if (cell->cellTypeData.sensor.autoTriggerInterval == 0) {
        if (!cell->signal.active) {
            return false;
        }
        if (cell->signal.active && abs(cell->signal.channels[0]) < TRIGGER_THRESHOLD) {
            return false;
        }
    } else {
        if (!cell->signal.active) {
            SignalProcessor::createEmptySignal(cell);
        }
        auto activationTime = max(MAX_SIGNAL_RELAXATION_TIME + 1, autoTriggerInterval);
        if (data.timestep % activationTime != 0) {
            return false;
        }
    }
    return true;
}
