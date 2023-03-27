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
    movement(SimulationData& data, SimulationResult& result, Cell* cell, Activity const& activity);
    __inline__ __device__ static void
    contractionExpansion(SimulationData& data, SimulationResult& result, Cell* cell, Activity const& activity);
    __inline__ __device__ static void
    bending(SimulationData& data, SimulationResult& result, Cell* cell, Activity const& activity);

    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    __inline__ __device__ static float getTruncatedValue(Activity const& activity, int channel = 0);
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
    auto activity = CellFunctionProcessor::calcInputActivity(cell);

    switch (cell->cellFunctionData.muscle.mode) {
    case MuscleMode_Movement: {
        movement(data, result, cell, activity);
    } break;
    case MuscleMode_ContractionExpansion: {
        contractionExpansion(data, result, cell, activity);
    } break;
    case MuscleMode_Bending: {
        bending(data, result, cell, activity);
    } break;
    }

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void
MuscleProcessor::movement(SimulationData& data, SimulationResult& result, Cell* cell, Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float2 direction = CellFunctionProcessor::calcSignalDirection(data, cell);
    if (direction.x != 0 || direction.y != 0) {
        cell->vel += Math::normalized(direction) * cudaSimulationParameters.cellFunctionMuscleMovementAcceleration[cell->color] * getTruncatedValue(activity);
    }
    cell->releaseLock();
    result.incMuscleActivity();
}

__device__ __inline__ void MuscleProcessor::contractionExpansion(
    SimulationData& data,
    SimulationResult& result,
    Cell* cell,
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
        if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            if (!connection.cell->tryLock()) {
                continue;
            }
            auto newDistance =
                connection.distance + cudaSimulationParameters.cellFunctionMuscleContractionExpansionDelta[cell->color] * getTruncatedValue(activity);
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
MuscleProcessor::bending(SimulationData& data, SimulationResult& result, Cell* cell, Activity const& activity)
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
        if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            auto intensityChannel0 = getTruncatedValue(activity);
            auto bendingAngle = cudaSimulationParameters.cellFunctionMuscleBendingAngle[cell->color] * intensityChannel0;

            if (bendingAngle < 0 && connection.angleFromPrevious <= -bendingAngle) {
                continue;
            }
            auto& nextConnection = cell->connections[(i + 1) % cell->numConnections];
            if (bendingAngle > 0 && nextConnection.angleFromPrevious <= bendingAngle) {
                continue;
            }
            connection.angleFromPrevious += bendingAngle;
            nextConnection.angleFromPrevious -= bendingAngle;
            if (cell->numConnections <= 2 && abs(activity.channels[1]) > cudaSimulationParameters.cellFunctionMuscleBendingAccelerationThreshold) {
                auto delta = Math::normalized(data.cellMap.getCorrectedDirection(connection.cell->absPos - cell->absPos));
                Math::rotateQuarterCounterClockwise(delta);
                auto intensityChannel1 = getTruncatedValue(activity, 1);
                if ((intensityChannel0 < -NEAR_ZERO && intensityChannel1 < -NEAR_ZERO) || (intensityChannel0 > NEAR_ZERO && intensityChannel1 > NEAR_ZERO)) {
                    MuscleBendingDirection bendingDirection = [&] {
                        if (intensityChannel0 > NEAR_ZERO) {
                            return MuscleBendingDirection_Positive;
                        } else if (intensityChannel0 < -NEAR_ZERO) {
                            return MuscleBendingDirection_Negative;
                        } else {
                            return MuscleBendingDirection_None;
                        }
                    }();
                    if (cell->cellFunctionData.muscle.lastBendingDirection == bendingDirection) {
                        ++cell->cellFunctionData.muscle.numConsecutiveBendings;
                    } else {
                        cell->cellFunctionData.muscle.numConsecutiveBendings = 0;
                    }
                    cell->cellFunctionData.muscle.lastBendingDirection = bendingDirection;
                    auto acceleration = delta * intensityChannel0 * cudaSimulationParameters.cellFunctionMuscleBendingAcceleration[cell->color]
                        * sqrtf(toFloat(cell->cellFunctionData.muscle.numConsecutiveBendings + 1)) / 20;

                    atomicAdd(&connection.cell->vel.x, acceleration.x);
                    atomicAdd(&connection.cell->vel.y, acceleration.y);
                }
            }
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

__inline__ __device__ float MuscleProcessor::getTruncatedValue(Activity const& activity, int channel)
{
    return max(-1.0f, min(1.0f, activity.channels[channel]));
}
