#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "Object.cuh"
#include "SimulationData.cuh"
#include "CellFunctionProcessor.cuh"
#include "SimulationStatistics.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity);
    __inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity);
    __inline__ __device__ static void bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity);

    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    __inline__ __device__ static bool hasTriangularConnection(Cell* cell, Cell* otherCell);
    __inline__ __device__ static float getTruncatedUnitValue(Activity const& activity, int channel = 0);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);

    switch (cell->cellFunctionData.muscle.mode) {
    case MuscleMode_Movement: {
        movement(data, statistics, cell, activity);
    } break;
    case MuscleMode_ContractionExpansion: {
        contractionExpansion(data, statistics, cell, activity);
    } break;
    case MuscleMode_Bending: {
        bending(data, statistics, cell, activity);
    } break;
    }

    CellFunctionProcessor::setActivity(cell, activity);
}

__device__ __inline__ void MuscleProcessor::movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    float2 direction = CellFunctionProcessor::calcSignalDirection(data, cell);
    float angle = max(-0.5f, min(0.5f, activity.channels[3])) * 360.0f;
    direction = Math::rotateClockwise(direction, angle);
    if (direction.x != 0 || direction.y != 0) {
        cell->vel += Math::normalized(direction) * cudaSimulationParameters.cellFunctionMuscleMovementAcceleration[cell->color] * getTruncatedUnitValue(activity);
    }
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
}

__device__ __inline__ void MuscleProcessor::contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity)
{
    if (abs(activity.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    auto const minDistance = cudaSimulationParameters.cellMinDistance * 1.2f;
    auto const maxDistance = max(cudaSimulationParameters.cellMaxBindingDistance * 0.5f, minDistance);
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            if (!connection.cell->tryLock()) {
                continue;
            }
            auto newDistance =
                connection.distance + cudaSimulationParameters.cellFunctionMuscleContractionExpansionDelta[cell->color] * getTruncatedUnitValue(activity);
            if (activity.channels[0] > 0 && newDistance >= maxDistance) {
                continue;
            }
            if (activity.channels[0] < 0 && newDistance <= minDistance) {
                continue;
            }
            connection.distance = newDistance;

            auto otherIndex = getConnectionIndex(connection.cell, cell);
            connection.cell->connections[otherIndex].distance = newDistance;
            connection.cell->releaseLock();
        }
    }
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
}

__inline__ __device__ void MuscleProcessor::bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Activity const& activity)
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
            auto intensityChannel0 = getTruncatedUnitValue(activity);
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

            MuscleBendingDirection bendingDirection = [&] {
                if (intensityChannel0 > NEAR_ZERO) {
                    return MuscleBendingDirection_Positive;
                } else if (intensityChannel0 < -NEAR_ZERO) {
                    return MuscleBendingDirection_Negative;
                } else {
                    return MuscleBendingDirection_None;
                }
            }();
            if (cell->cellFunctionData.muscle.lastBendingDirection == bendingDirection && cell->cellFunctionData.muscle.lastBendingSourceIndex == i) {
                cell->cellFunctionData.muscle.consecutiveBendingAngle += abs(bendingAngle);
            } else {
                cell->cellFunctionData.muscle.consecutiveBendingAngle = 0;
            }
            cell->cellFunctionData.muscle.lastBendingDirection = bendingDirection;
            cell->cellFunctionData.muscle.lastBendingSourceIndex = i;

            if (abs(activity.channels[1]) > cudaSimulationParameters.cellFunctionMuscleBendingAccelerationThreshold
                && !hasTriangularConnection(cell, connection.cell)) {
                auto delta = Math::normalized(data.cellMap.getCorrectedDirection(connection.cell->pos - cell->pos));
                Math::rotateQuarterCounterClockwise(delta);
                auto intensityChannel1 = getTruncatedUnitValue(activity, 1);
                if ((intensityChannel0 < -NEAR_ZERO && intensityChannel1 < -NEAR_ZERO) || (intensityChannel0 > NEAR_ZERO && intensityChannel1 > NEAR_ZERO)) {
                    auto acceleration = delta * intensityChannel0 * cudaSimulationParameters.cellFunctionMuscleBendingAcceleration[cell->color]
                        * sqrtf(cell->cellFunctionData.muscle.consecutiveBendingAngle + 1.0f) / 20 /*abs(bendingAngle) / 10*/;
                    atomicAdd(&connection.cell->vel.x, acceleration.x);
                    atomicAdd(&connection.cell->vel.y, acceleration.y);
                }
            }
        }
    }
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
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

__inline__ __device__ bool MuscleProcessor::hasTriangularConnection(Cell* cell, Cell* otherCell)
{
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell == otherCell) {
            continue;
        }
        for (int j = 0; j < connectedCell->numConnections; ++j) {
            auto connectedConnectedCell = connectedCell->connections[j].cell;
            if (connectedConnectedCell == otherCell) {
                return true;
            }
        }
    }
    return false;
}

__inline__ __device__ float MuscleProcessor::getTruncatedUnitValue(Activity const& activity, int channel)
{
    return max(-0.3f, min(0.3f, activity.channels[channel])) / 0.3f;
}
