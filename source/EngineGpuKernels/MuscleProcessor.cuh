#pragma once

#include "EngineInterface/CellTypeConstants.h"

#include "Object.cuh"
#include "SimulationData.cuh"
#include "SignalProcessor.cuh"
#include "SimulationStatistics.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    __inline__ __device__ static bool hasTriangularConnection(Cell* cell, Cell* otherCell);
    __inline__ __device__ static float getTruncatedUnitValue(Signal const& signal, int channel = 0);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    cell->cellTypeData.muscle.lastMovementX = 0;
    cell->cellTypeData.muscle.lastMovementY = 0;

    switch (cell->cellTypeData.muscle.mode) {
    case MuscleMode_Movement: {
        movement(data, statistics, cell);
    } break;
    case MuscleMode_ContractionExpansion: {
        contractionExpansion(data, statistics, cell);
    } break;
    case MuscleMode_Bending: {
        bending(data, statistics, cell);
    } break;
    }
}


__device__ __inline__ void MuscleProcessor::movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }

    auto direction = float2{0, 0};
    auto acceleration = 0.0f;
    if (cudaSimulationParameters.cellTypeMuscleMovementTowardTargetedObject) {
        if (cell->signal.origin == SignalOrigin_Sensor && (cell->signal.targetX != 0 || cell->signal.targetY != 0)) {
            direction = {cell->signal.targetX, cell->signal.targetY};
            acceleration = cudaSimulationParameters.cellTypeMuscleMovementAcceleration[cell->color];
        }
    } else {
        direction = SignalProcessor::calcReferenceDirection(data, cell);
        acceleration = cudaSimulationParameters.cellTypeMuscleMovementAcceleration[cell->color];
    }
    float angle = max(-0.5f, min(0.5f, cell->signal.channels[3])) * 360.0f;
    direction = Math::normalized(Math::rotateClockwise(direction, angle)) * acceleration * getTruncatedUnitValue(cell->signal);
    cell->vel += direction;
    cell->cellTypeData.muscle.lastMovementX = direction.x;
    cell->cellTypeData.muscle.lastMovementY = direction.y;
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
    radiate(data, cell);
}

__device__ __inline__ void MuscleProcessor::contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    auto const minDistance = cudaSimulationParameters.cellMinDistance * 1.2f;
    auto const maxDistance = max(cudaSimulationParameters.cellMaxBindingDistance[cell->color] * 0.5f, minDistance);
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        //if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            if (!connection.cell->tryLock()) {
                continue;
            }
            auto newDistance =
                connection.distance + cudaSimulationParameters.cellTypeMuscleContractionExpansionDelta[cell->color] * getTruncatedUnitValue(cell->signal);
            if (cell->signal.channels[0] > 0 && newDistance >= maxDistance) {
                continue;
            }
            if (cell->signal.channels[0] < 0 && newDistance <= minDistance) {
                continue;
            }
            connection.distance = newDistance;

            auto otherIndex = getConnectionIndex(connection.cell, cell);
            connection.cell->connections[otherIndex].distance = newDistance;
            connection.cell->releaseLock();
        //}
    }
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
    radiate(data, cell);
}

__inline__ __device__ void MuscleProcessor::bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
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
        //if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
        auto intensityChannel0 = getTruncatedUnitValue(cell->signal);
            auto bendingAngle = cudaSimulationParameters.cellTypeMuscleBendingAngle[cell->color] * intensityChannel0;

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
            if (cell->cellTypeData.muscle.lastBendingDirection == bendingDirection && cell->cellTypeData.muscle.lastBendingSourceIndex == i) {
                cell->cellTypeData.muscle.consecutiveBendingAngle += abs(bendingAngle);
            } else {
                cell->cellTypeData.muscle.consecutiveBendingAngle = 0;
            }
            cell->cellTypeData.muscle.lastBendingDirection = bendingDirection;
            cell->cellTypeData.muscle.lastBendingSourceIndex = i;

            if (abs(cell->signal.channels[1]) > TRIGGER_THRESHOLD && !hasTriangularConnection(cell, connection.cell)) {
                auto delta = Math::normalized(data.cellMap.getCorrectedDirection(connection.cell->pos - cell->pos));
                Math::rotateQuarterCounterClockwise(delta);
                auto intensityChannel1 = getTruncatedUnitValue(cell->signal, 1);
                if ((intensityChannel0 < -NEAR_ZERO && intensityChannel1 < -NEAR_ZERO) || (intensityChannel0 > NEAR_ZERO && intensityChannel1 > NEAR_ZERO)) {
                    auto acceleration = delta * intensityChannel0 * cudaSimulationParameters.cellTypeMuscleBendingAcceleration[cell->color]
                        * sqrtf(cell->cellTypeData.muscle.consecutiveBendingAngle + 1.0f) / 20 /*abs(bendingAngle) / 10*/;
                    atomicAdd(&connection.cell->vel.x, acceleration.x);
                    atomicAdd(&connection.cell->vel.y, acceleration.y);
                }
            }
        //}
    }
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
    radiate(data, cell);
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

__inline__ __device__ float MuscleProcessor::getTruncatedUnitValue(Signal const& signal, int channel)
{
    return max(-0.3f, min(0.3f, signal.channels[channel])) / 0.3f;
}

__inline__ __device__ void MuscleProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellTypeMuscleEnergyCost = cudaSimulationParameters.cellTypeMuscleEnergyCost[cell->color];
    if (cellTypeMuscleEnergyCost > 0) {
        RadiationProcessor::radiate(data, cell, cellTypeMuscleEnergyCost);
    }
}
