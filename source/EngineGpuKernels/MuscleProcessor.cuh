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

    //__inline__ __device__ static void movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    //__inline__ __device__ static void contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static float calcAngle(SimulationData& data, Cell* cell);

    //__inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static bool hasTriangularConnection(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static float getTruncatedUnitValue(Signal const& signal, int channel = 0);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);

    static auto constexpr ActivationCountdown = 20;
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
    //case MuscleMode_Movement: {
    //    movement(data, statistics, cell);
    //} break;
    //case MuscleMode_ContractionExpansion: {
    //    contractionExpansion(data, statistics, cell);
    //} break;
    case MuscleMode_Bending: {
        bending(data, statistics, cell);
    } break;
    }
}


//__device__ __inline__ void MuscleProcessor::movement(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
//        return;
//    }
//    if (!cell->tryLock()) {
//        return;
//    }
//
//    auto direction = SignalProcessor::calcReferenceDirection(data, cell);
//    auto acceleration = cudaSimulationParameters.cellTypeMuscleMovementAcceleration[cell->color];
//    float angle = max(-0.5f, min(0.5f, cell->signal.channels[3])) * 360.0f;
//    direction = Math::normalized(Math::rotateClockwise(direction, angle)) * acceleration * getTruncatedUnitValue(cell->signal);
//    cell->vel += direction;
//    cell->cellTypeData.muscle.lastMovementX = direction.x;
//    cell->cellTypeData.muscle.lastMovementY = direction.y;
//    cell->releaseLock();
//    statistics.incNumMuscleActivities(cell->color);
//    radiate(data, cell);
//}
//
//__device__ __inline__ void MuscleProcessor::contractionExpansion(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    if (abs(cell->signal.channels[0]) < NEAR_ZERO) {
//        return;
//    }
//    if (!cell->tryLock()) {
//        return;
//    }
//    auto const minDistance = cudaSimulationParameters.cellMinDistance * 1.2f;
//    auto const maxDistance = max(cudaSimulationParameters.cellMaxBindingDistance[cell->color] * 0.5f, minDistance);
//    for (int i = 0; i < cell->numConnections; ++i) {
//        auto& connection = cell->connections[i];
//        //if (connection.cell->executionOrderNumber == cell->inputExecutionOrderNumber) {
//            if (!connection.cell->tryLock()) {
//                continue;
//            }
//            auto newDistance =
//                connection.distance + cudaSimulationParameters.cellTypeMuscleContractionExpansionDelta[cell->color] * getTruncatedUnitValue(cell->signal);
//            if (cell->signal.channels[0] > 0 && newDistance >= maxDistance) {
//                continue;
//            }
//            if (cell->signal.channels[0] < 0 && newDistance <= minDistance) {
//                continue;
//            }
//            connection.distance = newDistance;
//
//            auto otherIndex = getConnectionIndex(connection.cell, cell);
//            connection.cell->connections[otherIndex].distance = newDistance;
//            connection.cell->releaseLock();
//        //}
//    }
//    cell->releaseLock();
//    statistics.incNumMuscleActivities(cell->color);
//    radiate(data, cell);
//}

__inline__ __device__ void MuscleProcessor::bending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.bending;

    if (cell->numConnections != 2) {
        return;
    }

    // Activation
    if (cell->signal.active) {
        bending.activation = max(-1.0f, min(1.0f, cell->signal.channels[0]));
        bending.activationCountdown = ActivationCountdown;
        if (abs(cell->signal.channels[1]) < SimulationParameters::cellTypeMuscleThreshold) {
            bending.bendingMode = BendingMode_BackAndForth;
        } else {
            bending.bendingMode = BendingMode_OneDirection;
        }
    }
    if (bending.activationCountdown == 0) {
        return;
    }

    // Initialization
    if (bending.initialAngle == 0) {
        bending.initialAngle = cell->connections[0].angleFromPrevious;
        auto orientation = Math::normalizedAngle(Math::subtractAngle(cell->absAngleToConnection0, muscle.frontAngle), -180.0f);
        if (orientation < 0) {
            bending.forward = false;
            bending.frontBackVelRatio = 1.0f - bending.frontBackVelRatio;
        }
        bending.lastAngle = calcAngle(data, cell);
        bending.impulseAlreadyApplied = true;
    }

    // Process auto bending
    if (SignalProcessor::isAutoTriggered(data, cell, 10)) {

        auto actualAngle = calcAngle(data, cell);
        auto activation = bending.activation * toFloat(bending.activationCountdown) / ActivationCountdown;

        if (bending.bendingMode == BendingMode_OneDirection) {
            auto forward = cell->signal.channels[1] >= 0;
            auto orientation = Math::normalizedAngle(Math::subtractAngle(cell->absAngleToConnection0, muscle.frontAngle), -180.0f);
            if (orientation < 0) {
                forward = !forward;
            }
            if (forward != bending.forward) {
                bending.impulseAlreadyApplied = false;
            }
            bending.forward = forward;
        }

        // Change direction
        bool skipBending = false;
        auto maxAngle = min(bending.initialAngle + bending.maxAngleDeviation, 300.0f);
        if (!bending.forward && cell->connections[0].angleFromPrevious > maxAngle) {
            if (bending.bendingMode == BendingMode_BackAndForth) {
                bending.forward = activation >= 0;
                bending.impulseAlreadyApplied = false;
            } else {
                skipBending = true;
            }
        }
        auto minAngle = max(bending.initialAngle - bending.maxAngleDeviation, 60.0f);
        if (bending.forward && cell->connections[0].angleFromPrevious < minAngle) {
            if (bending.bendingMode == BendingMode_BackAndForth) {
                bending.forward = activation < 0;
                bending.impulseAlreadyApplied = false;
            } else {
                skipBending = true;
            }
        }

        if (!skipBending) {
            auto angleDelta = bending.forward ? -(0.05f + bending.frontBackVelRatio) : 1.05f - bending.frontBackVelRatio;
            angleDelta *= 5.0f * activation;

            cell->connections[0].angleFromPrevious += angleDelta;
            cell->connections[1].angleFromPrevious -= angleDelta;

            // Apply impulse
            auto actualAngleDelta = actualAngle - bending.lastAngle;
            if (!bending.impulseAlreadyApplied) {
                if ((angleDelta < 0 && actualAngle < bending.initialAngle && cell->connections[0].angleFromPrevious < bending.initialAngle)
                    || (angleDelta > 0 && actualAngle > bending.initialAngle && cell->connections[0].angleFromPrevious > bending.initialAngle)) {
                    bending.impulseAlreadyApplied = true;

                    auto direction = Math::normalized(data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos));
                    if (actualAngleDelta < 0) {
                        Math::rotateQuarterClockwise(direction);
                    } else {
                        Math::rotateQuarterCounterClockwise(direction);
                    }
                    actualAngleDelta = min(5.0f, abs(actualAngleDelta) / 1);
                    if (bending.forward) {
                        actualAngleDelta *= bending.frontBackVelRatio;
                    } else {
                        actualAngleDelta *= 1.0f - bending.frontBackVelRatio;
                    }
                    auto acceleration =
                        direction * actualAngleDelta * actualAngleDelta * cudaSimulationParameters.cellTypeMuscleBendingAcceleration[cell->color] / 20.0f;

                    atomicAdd(&cell->connections[1].cell->vel.x, min(0.3f, max(-0.3f, acceleration.x)));
                    atomicAdd(&cell->connections[1].cell->vel.y, min(0.3f, max(-0.3f, acceleration.y)));
                }
            }
        }

        bending.lastAngle = actualAngle;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
        --bending.activationCountdown;
    }
}

__inline__ __device__ float MuscleProcessor::calcAngle(SimulationData& data, Cell* cell)
{
    auto direction0 = data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos);
    auto direction1 = data.cellMap.getCorrectedDirection(cell->connections[1].cell->pos - cell->pos);
    return Math::subtractAngle(Math::angleOfVector(direction0), Math::angleOfVector(direction1));
}

//__inline__ __device__ int MuscleProcessor::getConnectionIndex(Cell* cell, Cell* otherCell)
//{
//    for (int i = 0; i < cell->numConnections; ++i) {
//        if (cell->connections[i].cell == otherCell) {
//            return i;
//        }
//    }
//    return 0;
//}

//__inline__ __device__ bool MuscleProcessor::hasTriangularConnection(Cell* cell, Cell* otherCell)
//{
//    for (int i = 0; i < cell->numConnections; ++i) {
//        auto connectedCell = cell->connections[i].cell;
//        if (connectedCell == otherCell) {
//            continue;
//        }
//        for (int j = 0; j < connectedCell->numConnections; ++j) {
//            auto connectedConnectedCell = connectedCell->connections[j].cell;
//            if (connectedConnectedCell == otherCell) {
//                return true;
//            }
//        }
//    }
//    return false;
//}
//
//__inline__ __device__ float MuscleProcessor::getTruncatedUnitValue(Signal const& signal, int channel)
//{
//    return max(-0.3f, min(0.3f, signal.channels[channel])) / 0.3f;
//}
//
__inline__ __device__ void MuscleProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellTypeMuscleEnergyCost = cudaSimulationParameters.cellTypeMuscleEnergyCost[cell->color];
    if (cellTypeMuscleEnergyCost > 0) {
        RadiationProcessor::radiate(data, cell, cellTypeMuscleEnergyCost);
    }
}
