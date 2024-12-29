#pragma once

#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "SpotCalculator.cuh"

class SignalProcessor
{
public:
    __inline__ __device__ static void collectCellFunctionOperations(SimulationData& data);
    __inline__ __device__ static void calcFutureSignals(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static void createEmptySignal(Cell* cell);
    __inline__ __device__ static float2 calcSignalDirection(SimulationData& data, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void SignalProcessor::collectCellFunctionOperations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);

        if (cell->cellFunction != CellFunction_None) {
            if (cell->cellFunction == CellFunction_Detonator && cell->cellFunctionData.detonator.state == DetonatorState_Activated) {
                data.cellFunctionOperations[cell->cellFunction].tryAddEntry(CellFunctionOperation{cell});
            } else if (cell->livingState != LivingState_UnderConstruction && cell->livingState != LivingState_Activating && cell->activationTime == 0) {
                data.cellFunctionOperations[cell->cellFunction].tryAddEntry(CellFunctionOperation{cell});
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
            for (int k = 0, l = connectedCell->numSignalOrigins; k < l; ++k) {
                if (connectedCell->signalOrigins[k] == cell->id) {
                    skip = true;
                    break;
                }
            }
            if (skip) {
                continue;
            }
            cell->futureSignal.active = true;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                cell->futureSignal.channels[k] += connectedCell->signal.channels[k];
                cell->futureSignal.channels[k] = max(-10.0f, min(10.0f, cell->futureSignal.channels[k]));  // truncate value to avoid overflow
            }
            if (connectedCell->signal.origin == SignalOrigin_Sensor) {
                cell->futureSignal.origin = SignalOrigin_Sensor;
                cell->futureSignal.targetX += connectedCell->signal.targetX;
                cell->futureSignal.targetY += connectedCell->signal.targetY;
                ++numSensorSignals;
            }
            cell->futureSignalOrigins[numSignalOrigins] = connectedCell->id;
            ++numSignalOrigins;
        }
        cell->futureNumSignalOrigins = numSignalOrigins;
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
            cell->cellFunctionUsed = CellFunctionUsed_Yes;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->signal.channels[i] = cell->futureSignal.channels[i];
            }
            cell->signal.origin = cell->futureSignal.origin;
            cell->signal.targetX = cell->futureSignal.targetX;
            cell->signal.targetY = cell->futureSignal.targetY;
            cell->numSignalOrigins = cell->futureNumSignalOrigins;
            for (int i = 0; i < cell->numSignalOrigins; ++i) {
                cell->signalOrigins[i] = cell->futureSignalOrigins[i];
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
