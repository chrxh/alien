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
    __inline__ __device__ static void resetFetchedActivities(SimulationData& data);

    __inline__ __device__ static Activity calcInputActivity(Cell* cell);
    __inline__ __device__ static void setActivity(Cell* cell, Activity const& newActivity);
    __inline__ __device__ static void updateInvocationState(Cell* cell, Activity const& activity);

    struct ReferenceAndActualAngle
    {
        float referenceAngle;
        float actualAngle;
    };
    __inline__ __device__ static ReferenceAndActualAngle calcLargestGapReferenceAndActualAngle(SimulationData& data, Cell* cell, float angleDeviation);

    __inline__ __device__ static float2 calcSignalDirection(SimulationData& data, Cell* cell);

    __inline__ __device__ static int getCurrentExecutionNumber(SimulationData& data, Cell* cell);
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
        auto executionOrderNumber = getCurrentExecutionNumber(data, cell);

        if (cell->cellFunction != CellFunction_None && cell->executionOrderNumber == executionOrderNumber) {
            if (cell->cellFunction == CellFunction_Detonator && cell->cellFunctionData.detonator.state == DetonatorState_Activated) {
                data.cellFunctionOperations[cell->cellFunction].tryAddEntry(CellFunctionOperation{cell});
            } else if (cell->livingState == LivingState_Ready && cell->activationTime == 0) {
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

__inline__ __device__ void CellFunctionProcessor::resetFetchedActivities(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);
        if (cell->cellFunction == CellFunction_None) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->activity.channels[i] = 0;
            }
            continue;
        }
        auto executionOrderNumber = getCurrentExecutionNumber(data, cell);
        int maxOtherExecutionOrderNumber = -1;

        if (!cell->outputBlocked) {
            for (int i = 0, j = cell->numConnections; i < j; ++i) {
                auto otherExecutionOrderNumber = cell->connections[i].cell->executionOrderNumber;
                auto otherInputExecutionOrderNumber = cell->connections[i].cell->inputExecutionOrderNumber;
                auto otherOutputBlocked = cell->connections[i].cell->outputBlocked;
                bool flowToCell =
                    cell->inputExecutionOrderNumber == otherExecutionOrderNumber && !otherOutputBlocked
                      && cell->executionOrderNumber > otherInputExecutionOrderNumber;
                if (otherInputExecutionOrderNumber == cell->executionOrderNumber && !flowToCell) {
                    if (maxOtherExecutionOrderNumber == -1) {
                        maxOtherExecutionOrderNumber = otherExecutionOrderNumber;
                    } else {
                        if ((maxOtherExecutionOrderNumber > cell->executionOrderNumber
                             && (otherExecutionOrderNumber > maxOtherExecutionOrderNumber || otherExecutionOrderNumber < cell->executionOrderNumber))
                            || (maxOtherExecutionOrderNumber < cell->executionOrderNumber && otherExecutionOrderNumber > maxOtherExecutionOrderNumber
                                && otherExecutionOrderNumber < cell->executionOrderNumber)) {
                            maxOtherExecutionOrderNumber = otherExecutionOrderNumber;
                        }
                    }
                }
            }
        }
        if ((maxOtherExecutionOrderNumber == -1
             && executionOrderNumber == (cell->executionOrderNumber + 1) % cudaSimulationParameters.cellNumExecutionOrderNumbers)
            || (maxOtherExecutionOrderNumber != -1 && maxOtherExecutionOrderNumber == executionOrderNumber)) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                cell->activity.channels[i] = 0;
            }
        }
    }
}

__inline__ __device__ Activity CellFunctionProcessor::calcInputActivity(Cell* cell)
{
    Activity result;
    result.origin = ActivityOrigin_Unknown;
    result.targetX = 0;
    result.targetY = 0;

    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result.channels[i] = 0;
    }

    if (cell->inputExecutionOrderNumber == -1 || cell->inputExecutionOrderNumber == cell->executionOrderNumber) {
        return result;
    }

    for (int i = 0, j = cell->numConnections; i < j; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->outputBlocked || connectedCell->livingState != LivingState_Ready ) {
            continue;
        }
        if (connectedCell->inputExecutionOrderNumber == cell->executionOrderNumber && connectedCell->executionOrderNumber > cell->executionOrderNumber
            && !cell->outputBlocked) {
            continue;
        }
        if (connectedCell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                result.channels[i] += connectedCell->activity.channels[i];
                result.channels[i] = max(-1000000.0f, min(1000000.0f, result.channels[i])); //truncate value to avoid overflow
            }
            if (connectedCell->activity.origin == ActivityOrigin_Sensor) {
                result.origin = ActivityOrigin_Sensor;
                result.targetX = connectedCell->activity.targetX;
                result.targetY = connectedCell->activity.targetY;
            }
        }
    }
    return result;
}

__inline__ __device__ void CellFunctionProcessor::setActivity(Cell* cell, Activity const& newActivity)
{
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        cell->activity.channels[i] = newActivity.channels[i];
    }
    cell->activity.origin = newActivity.origin;
    cell->activity.targetX = newActivity.targetX;
    cell->activity.targetY = newActivity.targetY;
}

__inline__ __device__ void CellFunctionProcessor::updateInvocationState(Cell* cell, Activity const& activity)
{
    if (cell->cellFunctionUsed == CellFunctionUsed_No) {
        for (int i = 0; i < MAX_CHANNELS - 1; ++i) {
            if (activity.channels[i] != 0) {
                cell->cellFunctionUsed = CellFunctionUsed_Yes;
                break;
            }
        }
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
        if (connectedCell->executionOrderNumber == cell->inputExecutionOrderNumber && !connectedCell->outputBlocked) {
            auto directionDelta = cell->pos - connectedCell->pos;
            data.cellMap.correctDirection(directionDelta);
            result = result + Math::normalized(directionDelta);
        }
    }
    return Math::normalized(result);
}

__inline__ __device__ int CellFunctionProcessor::getCurrentExecutionNumber(SimulationData& data, Cell* cell)
{
    return toInt((data.timestep /*+ cell->creatureId*/) % cudaSimulationParameters.cellNumExecutionOrderNumbers);
}
