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
    __inline__ __device__ static void autoBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void manualBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void angleBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    //__inline__ __device__ static int getConnectionIndex(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static bool hasTriangularConnection(Cell* cell, Cell* otherCell);
    //__inline__ __device__ static float getTruncatedUnitValue(Signal const& signal, int channel = 0);
    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);

    struct BendingInfo
    {
        Cell* pivotCell;
        CellConnection* connection;
        CellConnection* connectionPrev;
        CellConnection* connectionNext;
    };
    __inline__ __device__ static BendingInfo getBendingInfo(SimulationData& data, Cell* cell);
    __inline__ __device__ static float calcActualAngle(SimulationData& data, BendingInfo const& bendingInfo);
    __inline__ __device__ static bool isCounterOriented(Cell* cell);

    static auto constexpr AccelerationLimit = 0.3f;
    static auto constexpr AutoTriggerInterval = 9;
    static auto constexpr AngleDistance = 30.0f;
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
    case MuscleMode_AutoBending: {
        autoBending(data, statistics, cell);
    } break;
    case MuscleMode_ManualBending: {
        manualBending(data, statistics, cell);
    } break;
    case MuscleMode_AngleBending: {
        angleBending(data, statistics, cell);
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

__inline__ __device__ void MuscleProcessor::autoBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.autoBending;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }

    // Activation
    if (cell->signal.active) {
        bending.activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleTrigger]));
        auto targetAngle = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::normalizedAngle(targetAngle + cell->angleToFront, -180.0f);

        auto angleFactor = [&] {
            if (isCounterOriented(cell)) {
                targetAngleRelToConnection0 = -targetAngleRelToConnection0;
            }
            if (cell->numConnections == 1) {
                targetAngleRelToConnection0 = Math::normalizedAngle(targetAngleRelToConnection0 + 180.0f, -180.0f);
            }
            if (targetAngleRelToConnection0 >= 0 && targetAngleRelToConnection0 < 90.0f) {
                return 0.0f;
            }
            if (targetAngleRelToConnection0 >= 90) {
                return 1.0f;
            }
            return min(1.0f, max(-1.0f, -targetAngleRelToConnection0 / 90.0f));
        }();
        bending.activation *= angleFactor * angleFactor * angleFactor;

        bending.activationCountdown = cudaSimulationParameters.cellTypeMuscleActivationCountdown;
    }
    if (bending.activationCountdown == 0) {
        return;
    }

    // Initialization
    if (bending.initialAngle == 0) {
        auto bendingInfo = getBendingInfo(data, cell);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        bending.forward = !isCounterOriented(cell);
        bending.lastActualAngle = calcActualAngle(data, bendingInfo);
        bending.impulseAlreadyApplied = true;
    }

    // Process auto bending
    if (SignalProcessor::isAutoTriggered(data, cell, AutoTriggerInterval)) {

        auto frontBackVelRatio = isCounterOriented(cell) ? 1.0f - bending.frontBackVelRatio : bending.frontBackVelRatio;

        auto bendingInfo = getBendingInfo(data, cell);
        auto actualAngle = calcActualAngle(data, bendingInfo);
        auto activation = bending.activation * toFloat(bending.activationCountdown) / cudaSimulationParameters.cellTypeMuscleActivationCountdown;

        // Change bending direction
        auto sumAngle = bendingInfo.connection->angleFromPrevious + bendingInfo.connectionNext->angleFromPrevious;  // Sum will not change
        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);

        if (bendingInfo.connection->angleFromPrevious /*actualAngle */ > maxAngle - NEAR_ZERO) {
            bending.forward = activation >= 0;
            bending.impulseAlreadyApplied = false;
        }
        if (bendingInfo.connection->angleFromPrevious /*actualAngle*/ < minAngle + NEAR_ZERO) {
            bending.forward = activation < 0;
            bending.impulseAlreadyApplied = false;
        }

        // Modify angle
        auto angleDelta = bending.forward ? -(0.05f + frontBackVelRatio) : 1.05f - frontBackVelRatio;
        angleDelta *= 5.0f * activation;

        if (bendingInfo.connection->angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - bendingInfo.connection->angleFromPrevious;
        }
        if (bendingInfo.connection->angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - bendingInfo.connection->angleFromPrevious;
        }
        bendingInfo.connection->angleFromPrevious += angleDelta;
        bendingInfo.connectionNext->angleFromPrevious -= angleDelta;

        // Apply impulse
        auto actualAngleDelta = actualAngle - bending.lastActualAngle;
        if (!bending.impulseAlreadyApplied) {
            if ((angleDelta < 0 && actualAngle < bending.initialAngle && bendingInfo.connection->angleFromPrevious < bending.initialAngle)
                || (angleDelta > 0 && actualAngle > bending.initialAngle && bendingInfo.connection->angleFromPrevious > bending.initialAngle)) {
                bending.impulseAlreadyApplied = true;

                auto direction = Math::normalized(data.cellMap.getCorrectedDirection(bendingInfo.connection->cell->pos - bendingInfo.pivotCell->pos));
                if (actualAngleDelta < 0) {
                    Math::rotateQuarterClockwise(direction);
                } else {
                    Math::rotateQuarterCounterClockwise(direction);
                }
                actualAngleDelta = min(5.0f, abs(actualAngleDelta));
                if (bending.forward) {
                    actualAngleDelta *= powf(frontBackVelRatio, 4.0f);
                } else {
                    actualAngleDelta *= powf(1.0f - frontBackVelRatio, 4.0f);
                }
                auto acceleration = direction * actualAngleDelta * cudaSimulationParameters.cellTypeMuscleBendingAcceleration[cell->color] / 1.5f;

                if (cell->numConnections == 2) {
                    int chainLength = 0;
                    Cell* connectedCell = cell;
                    while (connectedCell->numConnections == 2 && chainLength < 20) {
                        connectedCell = connectedCell->connections[1].cell;
                        ++chainLength;
                    }
                    connectedCell = cell;
                    float2 accPerCell{
                        min(AccelerationLimit, max(-AccelerationLimit, acceleration.x / chainLength)),
                        min(AccelerationLimit, max(-AccelerationLimit, acceleration.y / chainLength))};
                    while (connectedCell->numConnections == 2 && chainLength < 20) {
                        connectedCell = connectedCell->connections[1].cell;

                        atomicAdd(&connectedCell->vel.x, accPerCell.x);
                        atomicAdd(&connectedCell->vel.y, accPerCell.y);
                    }
                } else {
                    atomicAdd(&bendingInfo.pivotCell->vel.x, min(AccelerationLimit, max(-AccelerationLimit, acceleration.x)));
                    atomicAdd(&bendingInfo.pivotCell->vel.y, min(AccelerationLimit, max(-AccelerationLimit, acceleration.y)));
                }
            }
        }

        bending.lastActualAngle = actualAngle;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
        --bending.activationCountdown;
    }
}

__inline__ __device__ void MuscleProcessor::manualBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.manualBending;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }

    // Initialization
    if (bending.initialAngle == 0) {
        auto bendingInfo = getBendingInfo(data, cell);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        bending.lastActualAngle = calcActualAngle(data, bendingInfo);
        bending.impulseAlreadyApplied = true;
        bending.lastAngleDelta = 0;
    }

    // Process auto bending
    if (SignalProcessor::isManuallyTriggered(data, cell)) {

        auto bendingInfo = getBendingInfo(data, cell);
        auto actualAngle = calcActualAngle(data, bendingInfo);
        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleTrigger]));

        // Change bending direction
        auto sumAngle = bendingInfo.connection->angleFromPrevious + bendingInfo.connectionNext->angleFromPrevious;  // Sum will not change
        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);

        // Modify angle
        auto angleDelta = activation > 0 ? -(0.05f + bending.frontBackVelRatio) : -(1.05f - bending.frontBackVelRatio);
        angleDelta *= 5.0f * activation;
        if (isCounterOriented(cell)) {
            angleDelta = -angleDelta;
        }

        if (bendingInfo.connection->angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - bendingInfo.connection->angleFromPrevious;
        }
        if (bendingInfo.connection->angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - bendingInfo.connection->angleFromPrevious;
        }
        bendingInfo.connection->angleFromPrevious += angleDelta;
        bendingInfo.connectionNext->angleFromPrevious -= angleDelta;

        if ((angleDelta > 0 && bending.lastAngleDelta <= 0) || (angleDelta < 0 && bending.lastAngleDelta >= 0)) {
            bending.impulseAlreadyApplied = false;
        }
        bending.lastAngleDelta = angleDelta;

        // Apply impulse
        auto actualAngleDelta = actualAngle - bending.lastActualAngle;
        if (!bending.impulseAlreadyApplied) {
            if ((angleDelta < 0 && actualAngle < bending.initialAngle && bendingInfo.connection->angleFromPrevious < bending.initialAngle)
                || (angleDelta > 0 && actualAngle > bending.initialAngle && bendingInfo.connection->angleFromPrevious > bending.initialAngle)) {
                bending.impulseAlreadyApplied = true;

                auto direction = Math::normalized(data.cellMap.getCorrectedDirection(bendingInfo.connection->cell->pos - bendingInfo.pivotCell->pos));
                if (actualAngleDelta < 0) {
                    Math::rotateQuarterClockwise(direction);
                } else {
                    Math::rotateQuarterCounterClockwise(direction);
                }
                actualAngleDelta = min(5.0f, abs(actualAngleDelta));
                if (activation > 0) {
                    actualAngleDelta *= powf(bending.frontBackVelRatio, 4.0f);
                } else {
                    actualAngleDelta *= powf(1.0f - bending.frontBackVelRatio, 4.0f);
                }
                auto acceleration = direction * actualAngleDelta * cudaSimulationParameters.cellTypeMuscleBendingAcceleration[cell->color] / 1.5f;

                if (cell->numConnections == 2) {
                    int chainLength = 0;
                    Cell* connectedCell = cell;
                    while (connectedCell->numConnections == 2 && chainLength < 20) {
                        connectedCell = connectedCell->connections[1].cell;
                        ++chainLength;
                    }
                    connectedCell = cell;
                    float2 accPerCell{
                        min(AccelerationLimit, max(-AccelerationLimit, acceleration.x / chainLength)),
                        min(AccelerationLimit, max(-AccelerationLimit, acceleration.y / chainLength))};
                    while (connectedCell->numConnections == 2 && chainLength < 20) {
                        connectedCell = connectedCell->connections[1].cell;

                        atomicAdd(&connectedCell->vel.x, accPerCell.x);
                        atomicAdd(&connectedCell->vel.y, accPerCell.y);
                    }
                } else {
                    atomicAdd(&bendingInfo.pivotCell->vel.x, min(AccelerationLimit, max(-AccelerationLimit, acceleration.x)));
                    atomicAdd(&bendingInfo.pivotCell->vel.y, min(AccelerationLimit, max(-AccelerationLimit, acceleration.y)));
                }
            }
        }

        bending.lastActualAngle = actualAngle;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
}

__inline__ __device__ void MuscleProcessor::angleBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.angleBending;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }

    // Initialization
    if (bending.initialAngle == 0) {
        auto bendingInfo = getBendingInfo(data, cell);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
    }

    // Process auto bending
    if (SignalProcessor::isManuallyTriggered(data, cell)) {

        auto bendingInfo = getBendingInfo(data, cell);
        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleTrigger]));
        auto targetAngle = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::normalizedAngle(targetAngle + cell->angleToFront, -180.0f);

        // Change bending direction
        auto sumAngle = bendingInfo.connection->angleFromPrevious + bendingInfo.connectionNext->angleFromPrevious;  // Sum will not change
        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, AngleDistance), sumAngle - AngleDistance);

        // Modify angle
        auto angleDelta = activation > 0 ? (0.05f + bending.frontBackVelRatio) : (1.05f - bending.frontBackVelRatio);
        angleDelta *= 5.0f * activation;
        if (targetAngleRelToConnection0 < 0) {
            angleDelta = -angleDelta;
        }

        if (bendingInfo.connection->angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - bendingInfo.connection->angleFromPrevious;
        }
        if (bendingInfo.connection->angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - bendingInfo.connection->angleFromPrevious;
        }
        bendingInfo.connection->angleFromPrevious += angleDelta;
        bendingInfo.connectionNext->angleFromPrevious -= angleDelta;
        cell->angleToFront += angleDelta;

        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
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

__inline__ __device__ MuscleProcessor::BendingInfo MuscleProcessor::getBendingInfo(SimulationData& data, Cell* cell)
{
    BendingInfo result;
    if (cell->numConnections == 2) {
        result.pivotCell = cell;
        result.connection = &cell->connections[0];
        result.connectionPrev = &cell->connections[1];
        result.connectionNext = &cell->connections[1];
        return result;
    }

    // numConnections == 1
    else {
        auto privotCell = cell->connections[0].cell;
        result.pivotCell = privotCell;
        for (int i = 0; i < privotCell->numConnections; ++i) {
            if (privotCell->connections[i].cell == cell) {
                result.connection = &privotCell->connections[i];
                result.connectionPrev = &privotCell->connections[(i + privotCell->numConnections - 1) % privotCell->numConnections];
                result.connectionNext = &privotCell->connections[(i + 1) % privotCell->numConnections];
                break;
            }
        }
    }
    return result;
}

__inline__ __device__ float MuscleProcessor::calcActualAngle(SimulationData& data, BendingInfo const& bendingInfo)
{
    auto direction0 = data.cellMap.getCorrectedDirection(bendingInfo.connection->cell->pos - bendingInfo.pivotCell->pos);
    auto direction1 = data.cellMap.getCorrectedDirection(bendingInfo.connectionPrev->cell->pos - bendingInfo.pivotCell->pos);
    return Math::subtractAngle(Math::angleOfVector(direction0), Math::angleOfVector(direction1));
}

__inline__ __device__ bool MuscleProcessor::isCounterOriented(Cell* cell)
{
    if (cell->numConnections == 2) {
        return cell->angleToFront > NEAR_ZERO;
    } else {
        return cell->angleToFront < -NEAR_ZERO;
    }
}
