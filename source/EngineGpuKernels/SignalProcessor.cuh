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
        cell->futureSignal.origin = SignalOrigin_Unknown;
        cell->futureSignal.targetX = 0;
        cell->futureSignal.targetY = 0;

        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = 0;
        }

        int numSensorSignals = 0;
        int numSignalOrigins = 0;
        for (int i = 0, j = cell->numConnections; i < j; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->livingState == LivingState_UnderConstruction || !connectedCell->signal.active) {
                continue;
            }
            int skip = false;
            for (int k = 0, l = connectedCell->signal.numPrevCells; k < l; ++k) {
                if (connectedCell->signal.prevCellIds[k] == cell->id) {
                    skip = true;
                    break;
                }
            }
            if (skip) {
                continue;
            }

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
            cell->futureSignal.prevCellIds[numSignalOrigins] = connectedCell->id;
            ++numSignalOrigins;

        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = max(-10.0f, min(10.0f, cell->futureSignal.channels[i]));  // truncate value to avoid overflow
        }
        cell->futureSignal.numPrevCells = numSignalOrigins;
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
            cell->signal.numPrevCells = cell->futureSignal.numPrevCells;
            for (int i = 0; i < cell->signal.numPrevCells; ++i) {
                cell->signal.prevCellIds[i] = cell->futureSignal.prevCellIds[i];
            }
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
    cell->signal.numPrevCells = 0;
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
