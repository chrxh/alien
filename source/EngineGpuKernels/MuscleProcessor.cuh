#pragma once

#include "EngineInterface/CellFunctionEnums.h"

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"
#include "SimulationResult.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);

    __inline__ __device__ static void
    movement(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber, Activity const& activity);
    __inline__ __device__ static void
    contractionExpansion(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber, Activity const& activity);
    __inline__ __device__ static void
    bending(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber, Activity const& activity);

    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    __inline__ __device__ static float getIntensity(Activity const& activity);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

    switch (cell->cellFunctionData.muscle.mode) {
    case MuscleMode_Movement: {
        movement(data, result, cell, inputExecutionOrderNumber, activity);
    } break;
    case MuscleMode_ContractionExpansion: {
        contractionExpansion(data, result, cell, inputExecutionOrderNumber, activity);
    } break;
    case MuscleMode_Bending: {
        bending(data, result, cell, inputExecutionOrderNumber, activity);
    } break;
    }

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void
MuscleProcessor::movement(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber, Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float2 direction = CellFunctionProcessor::calcSignalDirection(data, cell, inputExecutionOrderNumber);
    if (direction.x != 0 || direction.y != 0) {
        cell->vel = cell->vel + Math::normalized(direction) * cudaSimulationParameters.cellFunctionMuscleMovementAcceleration * getIntensity(activity);
    }
    cell->releaseLock();
    result.incMuscleActivity();
}

__device__ __inline__ void MuscleProcessor::contractionExpansion(
    SimulationData& data,
    SimulationResult& result,
    Cell* cell,
    int const& inputExecutionOrderNumber,
    Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        if (connection.cell->executionOrderNumber == inputExecutionOrderNumber) {
            if (!connection.cell->tryLock()) {
                continue;
            }
            auto newDistance = connection.distance + cudaSimulationParameters.cellFunctionMuscleContractionExpansionDelta * getIntensity(activity);
            if (activity.channels[0] > 0 && newDistance >= cudaSimulationParameters.cellMaxBindingDistance * 0.8f) {
                continue;
            }
            if (activity.channels[0] < 0 && newDistance <= cudaSimulationParameters.cellMinDistance * 1.2f) {
                continue;
            }
            connection.distance = newDistance;

            auto otherIndex = getConnectionIndex(connection.cell, cell);
            connection.cell->connections[otherIndex].distance = newDistance;
            connection.cell->releaseLock();
        }
    }
    cell->releaseLock();
    result.incMuscleActivity();
}

__inline__ __device__ void
MuscleProcessor::bending(SimulationData& data, SimulationResult& result, Cell* cell, int const& inputExecutionOrderNumber, Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (cell->numConnections < 2) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float2 acceleration{0, 0};
    int numMatchingCells;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        if (connection.cell->executionOrderNumber == inputExecutionOrderNumber) {
            auto intensity = getIntensity(activity);
            auto bendingAngle = cudaSimulationParameters.cellFunctionMuscleBendingAngle * intensity;

            if (bendingAngle < 0 && connection.angleFromPrevious <= -bendingAngle) {
                continue;
            }
            auto& nextConnection = cell->connections[(i + 1) % cell->numConnections];
            if (bendingAngle > 0 && nextConnection.angleFromPrevious <= bendingAngle) {
                continue;
            }
            connection.angleFromPrevious += bendingAngle;
            nextConnection.angleFromPrevious -= bendingAngle;
            if (abs(activity.channels[1]) > cudaSimulationParameters.cellFunctionMuscleBendingAccelerationThreshold) {
                auto delta = Math::normalized(data.cellMap.getCorrectedDirection(connection.cell->absPos - cell->absPos));
                Math::rotateQuarterCounterClockwise(delta);
                delta = delta * intensity * cudaSimulationParameters.cellFunctionMuscleBendingAcceleration;
                acceleration = acceleration + delta;
                ++numMatchingCells;
            }
        }
    }
    if (numMatchingCells > 0) {
        atomicAdd(&cell->vel.x, acceleration.x / numMatchingCells);
        atomicAdd(&cell->vel.y, acceleration.y / numMatchingCells);
    }
    cell->releaseLock();
    result.incMuscleActivity();
}

__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == otherCell) {
            return i;
        }
    }
    return 0;
}

__inline__ __device__ float MuscleProcessor::getIntensity(Activity const& activity)
{
    return max(-1.0f, min(1.0f, activity.channels[0]));
}
