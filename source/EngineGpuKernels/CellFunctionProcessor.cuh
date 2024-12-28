#pragma once

#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "SpotCalculator.cuh"

class CellFunctionProcessor
{
public:
    __inline__ __device__ static void collectCellFunctionOperations(SimulationData& data);
    __inline__ __device__ static void updateRenderingData(SimulationData& data);
    __inline__ __device__ static void updateSignals(SimulationData& data);

    __inline__ __device__ static Signal updateFutureSignalOriginsAndReturnInputSignal(Cell* cell);
    __inline__ __device__ static void setSignal(Cell* cell, Signal const& newSignal);

    struct ReferenceAndActualAngle
    {
        float referenceAngle;
        float actualAngle;
    };
    __inline__ __device__ static ReferenceAndActualAngle calcLargestGapReferenceAndActualAngle(SimulationData& data, Cell* cell, float angleDeviation);

    __inline__ __device__ static float2 calcSignalDirection(SimulationData& data, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellFunctionProcessor::collectCellFunctionOperations(SimulationData& data)
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

__inline__ __device__ void CellFunctionProcessor::updateRenderingData(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->eventCounter > 0) {
            --cell->eventCounter;
        }
    }
}

__inline__ __device__ void CellFunctionProcessor::updateSignals(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        cell->signal.active = cell->futureSignal.active;
        if (cell->signal.active) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->signal.channels[i] = cell->futureSignal.channels[i];
            }
            cell->signal.origin = cell->futureSignal.origin;
            cell->signal.targetX = cell->futureSignal.targetX;
            cell->signal.targetY = cell->futureSignal.targetY;
        }
    }
}

__inline__ __device__ Signal CellFunctionProcessor::updateFutureSignalOriginsAndReturnInputSignal(Cell* cell)
{
    Signal result;
    result.active = false;
    result.origin = SignalOrigin_Unknown;
    result.targetX = 0;
    result.targetY = 0;

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.channels[i] = 0;
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
        result.active = true;
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            result.channels[k] += connectedCell->signal.channels[k];
            result.channels[k] = max(-10.0f, min(10.0f, result.channels[k])); // truncate value to avoid overflow
        }
        if (connectedCell->signal.origin == SignalOrigin_Sensor) {
            result.origin = SignalOrigin_Sensor;
            result.targetX += connectedCell->signal.targetX;
            result.targetY += connectedCell->signal.targetY;
            ++numSensorSignals;
        }
        cell->futureSignalOrigins[numSignalOrigins] = connectedCell->id;
        ++numSignalOrigins;
    }
    cell->futureNumSignalOrigins = numSignalOrigins;
    if (numSensorSignals > 0) {
        result.targetX /= toFloat(numSensorSignals);
        result.targetY /= toFloat(numSensorSignals);
    }
    return result;
}

__inline__ __device__ void CellFunctionProcessor::setSignal(Cell* cell, Signal const& newSignal)
{
    if (newSignal.active) {
        cell->cellFunctionUsed = CellFunctionUsed_Yes;
    }
    cell->futureSignal.active = newSignal.active;
    if (newSignal.active) {
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            cell->futureSignal.channels[i] = newSignal.channels[i];
        }
        cell->futureSignal.origin = newSignal.origin;
        cell->futureSignal.targetX = newSignal.targetX;
        cell->futureSignal.targetY = newSignal.targetY;
    }
}


__inline__ __device__ CellFunctionProcessor::ReferenceAndActualAngle
CellFunctionProcessor::calcLargestGapReferenceAndActualAngle(SimulationData& data, Cell* cell, float angleDeviation)
{
    if (0 == cell->numConnections) {
        return ReferenceAndActualAngle{0, data.numberGen1.random()*360};
    }
    auto displacement = cell->connections[0].cell->pos - cell->pos;
    data.cellMap.correctDirection(displacement);
    auto angle = Math::angleOfVector(displacement);
    int index = 0;
    float largestAngleGap = 0;
    float angleOfLargestAngleGap = 0;
    auto numConnections = cell->numConnections;
    for (int i = 1; i <= numConnections; ++i) {
        auto angleDiff = cell->connections[i % numConnections].angleFromPrevious;
        if (angleDiff > largestAngleGap) {
            largestAngleGap = angleDiff;
            index = i % numConnections;
            angleOfLargestAngleGap = angle;
        }
        angle += angleDiff;
    }

    auto angleFromPrev = cell->connections[index].angleFromPrevious;
    for (int i = 0; i < numConnections - 1; ++i) {
        if (angleDeviation > angleFromPrev / 2) {
            angleDeviation -= angleFromPrev / 2;
            index = (index + 1) % numConnections;
            angleOfLargestAngleGap += angleFromPrev; 
            angleFromPrev = cell->connections[index].angleFromPrevious;
            angleDeviation = angleDeviation - angleFromPrev / 2;
        }
        if (angleDeviation < -angleFromPrev / 2) {
            angleDeviation += angleFromPrev / 2;
            index = (index + numConnections - 1) % numConnections;
            angleFromPrev = cell->connections[index].angleFromPrevious;
            angleDeviation = angleDeviation + angleFromPrev / 2;
            angleOfLargestAngleGap -= angleFromPrev;
        }
    }
    auto angleFromPreviousConnection = angleFromPrev / 2 + angleDeviation;

    if (angleFromPreviousConnection > 360.0f) {
        angleFromPreviousConnection -= 360;
    }
    angleFromPreviousConnection = max(30.0f, min(angleFromPrev - 30.0f, angleFromPreviousConnection));

    return ReferenceAndActualAngle{angleFromPreviousConnection, angleOfLargestAngleGap + angleFromPreviousConnection};
}

__inline__ __device__ float2 CellFunctionProcessor::calcSignalDirection(SimulationData& data, Cell* cell)
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
