#pragma once

#include "EngineInterface/Enums.h"

#include "Cell.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"

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
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

    if (cell->cellFunctionData.muscle.mode == Enums::MuscleMode_Movement) {
        movement(data, result, cell, inputExecutionOrderNumber, activity);
    }

    if (cell->cellFunctionData.muscle.mode == Enums::MuscleMode_ContractionExpansion) {
        contractionExpansion(data, result, cell, inputExecutionOrderNumber, activity);
    }

    if (cell->cellFunctionData.muscle.mode == Enums::MuscleMode_Bending) {
        bending(data, result, cell, inputExecutionOrderNumber, activity);
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
    float2 direction{0, 0};
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connectedCell = cell->connections[i].cell;
        if (connectedCell->executionOrderNumber == inputExecutionOrderNumber) {
            if (!connectedCell->tryLock()) {
                continue;
            }
            auto directionDelta = cell->absPos - connectedCell->absPos;
            data.cellMap.correctDirection(directionDelta);
            direction = direction + Math::normalized(directionDelta);
            connectedCell->releaseLock();
        }
    }
    if (direction.x != 0 || direction.y != 0) {
        cell->vel = cell->vel + Math::normalized(direction) * cudaSimulationParameters.cellFunctionMuscleMovementDelta * getIntensity(activity);
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
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        if (connection.cell->executionOrderNumber == inputExecutionOrderNumber) {
            auto intensity = cudaSimulationParameters.cellFunctionMuscleBendingAngle * getIntensity(activity);

            if (intensity < 0 && connection.angleFromPrevious <= -intensity) {
                continue;
            }
            auto& nextConnection = cell->connections[(i + 1) % cell->numConnections];
            if (intensity > 0 && nextConnection.angleFromPrevious <= intensity) {
                continue;
            }
            connection.angleFromPrevious += intensity;
            nextConnection.angleFromPrevious -= intensity;
        }
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
