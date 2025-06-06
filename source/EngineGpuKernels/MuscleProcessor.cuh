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

    __inline__ __device__ static void movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal);
    __inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal const& signal);
    __inline__ __device__ static void bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal const& signal);

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
    auto& operations = data.cellFunctionOperations[CellFunction_Muscle];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto signal = CellFunctionProcessor::calcInputSignal(cell);
    CellFunctionProcessor::updateInvocationState(cell, signal);

    cell->cellFunctionData.muscle.lastMovementX = 0;
    cell->cellFunctionData.muscle.lastMovementY = 0;

    switch (cell->cellFunctionData.muscle.mode) {
    case MuscleMode_Movement: {
        movement(data, statistics, cell, signal);
    } break;
    case MuscleMode_ContractionExpansion: {
        contractionExpansion(data, statistics, cell, signal);
    } break;
    case MuscleMode_Bending: {
        bending(data, statistics, cell, signal);
    } break;
    }

    CellFunctionProcessor::setSignal(cell, signal);
}


namespace
{
    __device__ Cell* findNearbySensor(Cell* cell)
    {
        for (int i = 0, j = cell->numConnections; i < j; ++i) {
            auto const& connectedCell = cell->connections[i].cell;
            if (connectedCell->cellFunction == CellFunction_Sensor) {
                return connectedCell;
            }
        }
        for (int i = 0, j = cell->numConnections; i < j; ++i) {
            auto const& connectedCell = cell->connections[i].cell;
            for (int k = 0, l = connectedCell->numConnections; k < l; ++k) {
                auto const& connectedConnectedCell = connectedCell->connections[k].cell;
                if (connectedConnectedCell->cellFunction == CellFunction_Sensor) {
                    return connectedConnectedCell;
                }
            }
        }
        return nullptr;
    }
}

__device__ __inline__ void MuscleProcessor::movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal)
{
    if (abs(signal.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }

    auto direction = float2{0, 0};
    auto acceleration = 0.0f;
    if (cudaSimulationParameters.cellFunctionMuscleMovementTowardTargetedObject) {
        if (cudaSimulationParameters.features.legacyModes && cudaSimulationParameters.legacyCellFunctionMuscleMovementAngleFromSensor) {
            if (auto sensorCell = findNearbySensor(cell)) {
                auto const& sensorData = sensorCell->cellFunctionData.sensor;
                if (sensorData.memoryTargetX != 0 || sensorData.memoryTargetY != 0) {
                    direction = {sensorData.memoryTargetX, sensorData.memoryTargetY};
                    acceleration = cudaSimulationParameters.cellFunctionMuscleMovementAcceleration[cell->color];
                }
            }
        } else {
            if (signal.origin == SignalOrigin_Sensor && (signal.targetX != 0 || signal.targetY != 0)) {
                direction = {signal.targetX, signal.targetY};
                acceleration = cudaSimulationParameters.cellFunctionMuscleMovementAcceleration[cell->color];
            }
        }
    } else {
        direction = CellFunctionProcessor::calcSignalDirection(data, cell);
        acceleration = cudaSimulationParameters.cellFunctionMuscleMovementAcceleration[cell->color];
    }
    float angle = max(-0.5f, min(0.5f, signal.channels[3])) * 360.0f;
    direction = Math::normalized(Math::rotateClockwise(direction, angle)) * acceleration * getTruncatedUnitValue(signal);
    cell->vel += direction;
    cell->cellFunctionData.muscle.lastMovementX = direction.x;
    cell->cellFunctionData.muscle.lastMovementY = direction.y;
    cell->releaseLock();
    statistics.incNumMuscleActivities(cell->color);
    radiate(data, cell);
}

__device__ __inline__ void MuscleProcessor::contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal const& signal)
{
    if (abs(signal.channels[0]) < NEAR_ZERO) {
        return;
    }
    if (!cell->tryLock()) {
        return;
    }
    auto const minDistance = cudaSimulationParameters.cellMinDistance * 1.2f;
    auto const maxDistance = max(cudaSimulationParameters.cellMaxBindingDistance[cell->color] * 0.5f, minDistance);
    for (int i = 0; i < cell->numConnections; ++i) {
        auto& connection = cell->connections[i];
        if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
            if (!connection.cell->tryLock()) {
                continue;
            }
            auto newDistance =
                connection.distance + cudaSimulationParameters.cellFunctionMuscleContractionExpansionDelta[cell->color] * getTruncatedUnitValue(signal);
            if (signal.channels[0] > 0 && newDistance >= maxDistance) {
                continue;
            }
            if (signal.channels[0] < 0 && newDistance <= minDistance) {
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
    radiate(data, cell);
}

__inline__ __device__ void MuscleProcessor::bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal const& signal)
{
    if (abs(signal.channels[0]) < NEAR_ZERO) {
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
            auto intensityChannel0 = getTruncatedUnitValue(signal);
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

            if (abs(signal.channels[1]) > cudaSimulationParameters.cellFunctionMuscleBendingAccelerationThreshold
                && !hasTriangularConnection(cell, connection.cell)) {
                auto delta = Math::normalized(data.cellMap.getCorrectedDirection(connection.cell->pos - cell->pos));
                Math::rotateQuarterCounterClockwise(delta);
                auto intensityChannel1 = getTruncatedUnitValue(signal, 1);
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
    auto cellFunctionMuscleEnergyCost = cudaSimulationParameters.cellFunctionMuscleEnergyCost[cell->color];
    if (cellFunctionMuscleEnergyCost > 0) {
        RadiationProcessor::radiate(data, cell, cellFunctionMuscleEnergyCost);
    }
}
