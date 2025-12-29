#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

    __inline__ __device__ static float getInitialAngleFromPrevious(Cell* cell, int connectionIndex);  // Return the angleFromPrevious without muscle distortions
    __inline__ __device__ static void restoreInitialAngleFromPrevious(Cell* muscleCell, Cell* affectedCell, int connectionIndex);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void autoBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void manualBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void angleBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void autoCrawling(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void manualCrawling(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void directMovement(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void restoreInitialAngleFromPreviousIntern(float& initialAngle, Cell* muscleCell, Cell* affectedCell, int connectionIndex);

    __inline__ __device__ static void radiate(SimulationData& data, Cell* cell);

    __inline__ __device__ static void getChain(Cell** chain, int& chainLength, Cell* startCell);
    __inline__ __device__ static float2 calcAverageDirection(SimulationData& data, Cell* cell);
    __inline__ __device__ static void applyAcceleration(Cell* cell, float2 const& acceleration);

    struct BendingInfo
    {
        Cell* pivotCell;
        CellConnection* connection;
        CellConnection* connectionPrev;
        CellConnection* connectionNext;
    };
    __inline__ __device__ static BendingInfo getBendingInfo(Cell* cell);

    __inline__ __device__ static bool isLeftSide(Cell* cell);

    static auto constexpr AccelerationLimit = 0.3f;
    static auto constexpr AutoTriggerInterval = 9;
    static auto constexpr MinAngle = 30.0f;
    static auto constexpr MinDistance = 0.1f;
    static auto constexpr MaxChainLength = 5;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void MuscleProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Muscle];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__device__ __inline__ float MuscleProcessor::getInitialAngleFromPrevious(Cell* cell, int connectionIndex)
{
    CUDA_CHECK(connectionIndex < cell->numConnections);
    for (int i = 0, j = cell->numConnections; i < j; ++i) {
        auto connectedCell = cell->connections[i].cell;
        if (connectedCell->cellType == CellType_Muscle
            && (connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedCell->cellTypeData.muscle.mode == MuscleMode_ManualBending)) {

            auto const& initialAngle = connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending
                ? connectedCell->cellTypeData.muscle.modeData.autoBending.initialAngle
                : connectedCell->cellTypeData.muscle.modeData.manualBending.initialAngle;
            if (initialAngle != VALUE_NOT_SET_FLOAT) {
                auto bendingInfo = MuscleProcessor::getBendingInfo(connectedCell);
                if (bendingInfo.connection == &cell->connections[connectionIndex]) {
                    return initialAngle;
                }
                if (bendingInfo.connectionNext == &cell->connections[connectionIndex]) {
                    return bendingInfo.connectionNext->angleFromPrevious - (initialAngle - bendingInfo.connection->angleFromPrevious);
                }
            }
        }
    }
    return cell->connections[connectionIndex].angleFromPrevious;
}

__device__ __inline__ void MuscleProcessor::restoreInitialAngleFromPrevious(Cell* muscleCell, Cell* affectedCell, int connectionIndex)
{
    auto& muscle = muscleCell->cellTypeData.muscle;
    if (muscle.mode == MuscleMode_AutoBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.autoBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    } else if (muscle.mode == MuscleMode_ManualBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.manualBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    } else if (muscle.mode == MuscleMode_AngleBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.angleBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    cell->cellTypeData.muscle.lastMovementX = 0;
    cell->cellTypeData.muscle.lastMovementY = 0;

    switch (cell->cellTypeData.muscle.mode % MuscleMode_Count) {
    case MuscleMode_AutoBending: {
        autoBending(data, statistics, cell);
    } break;
    case MuscleMode_ManualBending: {
        manualBending(data, statistics, cell);
    } break;
    case MuscleMode_AngleBending: {
        angleBending(data, statistics, cell);
    } break;
    case MuscleMode_AutoCrawling: {
        autoCrawling(data, statistics, cell);
    } break;
    case MuscleMode_ManualCrawling: {
        manualCrawling(data, statistics, cell);
    } break;
    case MuscleMode_DirectMovement: {
        directMovement(data, statistics, cell);
    } break;
    }
}

__inline__ __device__ void MuscleProcessor::autoBending(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& bending = muscle.modeData.autoBending;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }
    // Activation
    if (cell->signalState == SignalState_Active) {
        bending.activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));
        auto targetAngle = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::getNormalizedAngle(targetAngle + cell->frontAngle, -180.0f);

        auto angleFactor = [&] {
            if (isLeftSide(cell)) {
                targetAngleRelToConnection0 = -targetAngleRelToConnection0;
            }
            if (cell->numConnections == 1) {
                targetAngleRelToConnection0 = Math::getNormalizedAngle(targetAngleRelToConnection0 + 180.0f, -180.0f);
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

        bending.activationCountdown = cudaSimulationParameters.muscleActivationCountdown;
    }
    if (bending.activationCountdown == 0) {
        return;
    }

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(cell);

        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        auto pivotCell = bendingInfo.pivotCell;
        for (int k = 0, l = pivotCell->numConnections; k < l; ++k) {
            auto connectedCell = pivotCell->connections[k].cell;
            if (connectedCell->cellType == CellType_Muscle
                && (connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedCell->cellTypeData.muscle.mode == MuscleMode_ManualBending)) {
                auto const& initialAngle = connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending
                    ? connectedCell->cellTypeData.muscle.modeData.autoBending.initialAngle
                    : connectedCell->cellTypeData.muscle.modeData.manualBending.initialAngle;
                if (initialAngle != VALUE_NOT_SET_FLOAT) {
                    auto connectedBendingInfo = MuscleProcessor::getBendingInfo(connectedCell);
                    if (connectedBendingInfo.connection == bendingInfo.connection) {
                        bending.initialAngle = initialAngle;
                        break;
                    }
                }
            }
        }

        bending.forward = !isLeftSide(cell);
        bending.impulseAlreadyApplied = true;
    }

    // Process auto bending
    if (SignalProcessor::isAutoTriggered(data, cell, AutoTriggerInterval)) {

        auto forwardBackwardRatio = isLeftSide(cell) ? bending.forwardBackwardRatio : 1.0f - bending.forwardBackwardRatio;

        auto bendingInfo = getBendingInfo(cell);
        auto activation = bending.activation * toFloat(bending.activationCountdown) / cudaSimulationParameters.muscleActivationCountdown;

        // Change bending direction
        auto angleFromPrevious = alienAtomicRead(&bendingInfo.connection->angleFromPrevious);
        auto angleFromPrevious2 = alienAtomicRead(&bendingInfo.connectionNext->angleFromPrevious);
        auto sumAngle = angleFromPrevious + angleFromPrevious2;  // Sum will not change

        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, MinAngle), sumAngle - MinAngle);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, MinAngle), sumAngle - MinAngle);

        if (angleFromPrevious > maxAngle - NEAR_ZERO) {
            bending.forward = activation >= 0;
            bending.impulseAlreadyApplied = false;
        }
        if (angleFromPrevious < minAngle + NEAR_ZERO) {
            bending.forward = activation < 0;
            bending.impulseAlreadyApplied = false;
        }

        // Modify angle
        auto angleDelta = bending.forward ? -(0.05f + forwardBackwardRatio) : 1.05f - forwardBackwardRatio;
        angleDelta *= 5.0f * activation;

        if (angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - angleFromPrevious;
        }
        if (angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - angleFromPrevious;
        }
        atomicAdd(&bendingInfo.connection->angleFromPrevious, angleDelta);
        atomicAdd(&bendingInfo.connectionNext->angleFromPrevious, -angleDelta);

        // Apply impulse
        if (!bending.impulseAlreadyApplied) {
            angleFromPrevious = alienAtomicRead(&bendingInfo.connection->angleFromPrevious);
            if ((angleDelta < 0 && angleFromPrevious < bending.initialAngle) || (angleDelta > 0 && angleFromPrevious > bending.initialAngle)) {
                bending.impulseAlreadyApplied = true;

                auto direction = calcAverageDirection(data, cell);

                if (angleDelta < 0) {
                    Math::rotateQuarterClockwise(direction);
                } else {
                    Math::rotateQuarterCounterClockwise(direction);
                }
                angleDelta = min(5.0f, abs(angleDelta));
                if (bending.forward) {
                    angleDelta *= powf(forwardBackwardRatio, 4.0f);
                } else {
                    angleDelta *= powf(1.0f - forwardBackwardRatio, 4.0f);
                }
                auto acceleration = direction * angleDelta * cudaSimulationParameters.muscleBendingAcceleration.value[cell->color] / 9.0f;
                applyAcceleration(cell, acceleration);
            }
        }

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
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(cell);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        auto pivotCell = bendingInfo.pivotCell;
        for (int k = 0, l = pivotCell->numConnections; k < l; ++k) {
            auto connectedCell = pivotCell->connections[k].cell;
            if (connectedCell->cellType == CellType_Muscle
                && (connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedCell->cellTypeData.muscle.mode == MuscleMode_ManualBending)) {
                auto const& initialAngle = connectedCell->cellTypeData.muscle.mode == MuscleMode_AutoBending
                    ? connectedCell->cellTypeData.muscle.modeData.autoBending.initialAngle
                    : connectedCell->cellTypeData.muscle.modeData.manualBending.initialAngle;
                if (initialAngle != VALUE_NOT_SET_FLOAT) {
                    auto connectedBendingInfo = MuscleProcessor::getBendingInfo(connectedCell);
                    if (connectedBendingInfo.connection == bendingInfo.connection) {
                        bending.initialAngle = initialAngle;
                        break;
                    }
                }
            }
        }

        bending.impulseAlreadyApplied = true;
        bending.lastAngleDelta = 0;
    }

    // Process manual bending
    if (SignalProcessor::isManuallyTriggered(data, cell)) {

        auto bendingInfo = getBendingInfo(cell);
        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));

        // Change bending direction
        auto angleFromPrevious = alienAtomicRead(&bendingInfo.connection->angleFromPrevious);
        auto angleFromPrevious2 = alienAtomicRead(&bendingInfo.connectionNext->angleFromPrevious);
        auto sumAngle = angleFromPrevious + angleFromPrevious2;  // Sum will not change

        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, MinAngle), sumAngle - MinAngle);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, MinAngle), sumAngle - MinAngle);

        // Modify angle
        auto angleDelta = activation > 0 ? -1.05f + bending.forwardBackwardRatio : -0.05f - bending.forwardBackwardRatio;
        angleDelta *= 5.0f * activation;
        if (isLeftSide(cell)) {
            angleDelta = -angleDelta;
        }

        if (angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - angleFromPrevious;
        }
        if (angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - angleFromPrevious;
        }
        atomicAdd(&bendingInfo.connection->angleFromPrevious, angleDelta);
        atomicAdd(&bendingInfo.connectionNext->angleFromPrevious, -angleDelta);

        if ((angleDelta > 0 && bending.lastAngleDelta <= 0) || (angleDelta < 0 && bending.lastAngleDelta >= 0)) {
            bending.impulseAlreadyApplied = false;
        }
        bending.lastAngleDelta = angleDelta;

        // Apply impulse
        if (!bending.impulseAlreadyApplied) {
            angleFromPrevious = alienAtomicRead(&bendingInfo.connection->angleFromPrevious);
            if ((angleDelta < 0 && angleFromPrevious < bending.initialAngle) || (angleDelta > 0 && angleFromPrevious > bending.initialAngle)) {
                bending.impulseAlreadyApplied = true;

                auto direction = calcAverageDirection(data, cell);

                if (angleDelta < 0) {
                    Math::rotateQuarterClockwise(direction);
                } else {
                    Math::rotateQuarterCounterClockwise(direction);
                }
                angleDelta = min(5.0f, abs(angleDelta));
                if (activation > 0) {
                    angleDelta *= powf(1.0f - bending.forwardBackwardRatio, 4.0f);
                } else {
                    angleDelta *= powf(bending.forwardBackwardRatio, 4.0f);
                }
                auto acceleration = direction * angleDelta * cudaSimulationParameters.muscleBendingAcceleration.value[cell->color] / 9.0f;
                applyAcceleration(cell, acceleration);
            }
        }

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
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(cell);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
    }

    // Process angle bending
    if (SignalProcessor::isManuallyTriggered(data, cell)) {

        auto bendingInfo = getBendingInfo(cell);
        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));
        auto targetAngle = max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::getNormalizedAngle(cell->frontAngle + targetAngle, -180.0f);

        // Change bending direction
        auto angleFromPrevious = alienAtomicRead(&bendingInfo.connection->angleFromPrevious);
        auto angleFromPrevious2 = alienAtomicRead(&bendingInfo.connectionNext->angleFromPrevious);
        auto sumAngle = angleFromPrevious + angleFromPrevious2;  // Sum will not change
        auto maxAngleDeviation = min(bending.initialAngle, sumAngle - bending.initialAngle) * bending.maxAngleDeviation / 2;
        auto maxAngle = min(max(bending.initialAngle + maxAngleDeviation, MinAngle), sumAngle - MinAngle);
        auto minAngle = min(max(bending.initialAngle - maxAngleDeviation, MinAngle), sumAngle - MinAngle);

        // Modify angle
        auto angleDelta = activation > 0 ? (1.05f - bending.attractionRepulsionRatio) : (0.05f + bending.attractionRepulsionRatio);
        angleDelta *= 5.0f * activation;
        if (targetAngleRelToConnection0 < 0) {
            angleDelta = -angleDelta;
        }
        if (cell->numConnections == 1) {
            angleDelta = -angleDelta;
        }

        if (angleFromPrevious + angleDelta > maxAngle) {
            angleDelta = maxAngle - angleFromPrevious;
        }
        if (angleFromPrevious + angleDelta < minAngle) {
            angleDelta = minAngle - angleFromPrevious;
        }
        atomicAdd(&bendingInfo.connection->angleFromPrevious, angleDelta);
        atomicAdd(&bendingInfo.connectionNext->angleFromPrevious, -angleDelta);
        cell->frontAngle -= angleDelta;

        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
}

__inline__ __device__ void MuscleProcessor::autoCrawling(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& crawling = muscle.modeData.autoCrawling;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Activation
    if (cell->signalState == SignalState_Active) {
        crawling.activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));
        crawling.activationCountdown = cudaSimulationParameters.muscleActivationCountdown;
    }
    if (crawling.activationCountdown == 0) {
        return;
    }

    // Initialization
    if (crawling.initialDistance == VALUE_NOT_SET_FLOAT) {
        crawling.initialDistance = cell->connections[0].distance;
        crawling.forward = true;
        crawling.lastActualDistance = data.cellMap.getDistance(cell->connections[0].cell->pos, cell->pos);
        crawling.impulseAlreadyApplied = true;
    }

    // Process auto crawling
    if (SignalProcessor::isAutoTriggered(data, cell, AutoTriggerInterval)) {

        auto actualDistance = data.cellMap.getDistance(cell->connections[0].cell->pos, cell->pos);
        auto activation = crawling.activation * toFloat(crawling.activationCountdown) / cudaSimulationParameters.muscleActivationCountdown;

        // Change crawling direction
        auto maxDistanceDeviation = max(0.0f, min(1.0f, crawling.maxDistanceDeviation));
        auto maxDistance = max(crawling.initialDistance * (1.0f + maxDistanceDeviation), MinDistance);
        auto minDistance = min(max(crawling.initialDistance * (1.0f - maxDistanceDeviation), MinDistance), crawling.initialDistance);

        if (cell->connections[0].distance > maxDistance - NEAR_ZERO) {
            crawling.forward = activation >= 0;
            crawling.impulseAlreadyApplied = false;
        }
        if (cell->connections[0].distance < minDistance + NEAR_ZERO) {
            crawling.forward = activation < 0;
            crawling.impulseAlreadyApplied = false;
        }

        // Calc and apply distance delta
        auto distanceDelta = crawling.forward ? -1.05f + crawling.forwardBackwardRatio : 0.05f + crawling.forwardBackwardRatio;
        distanceDelta *= 0.25f * activation;
        if (cell->connections[0].distance + distanceDelta > maxDistance) {
            distanceDelta = maxDistance - cell->connections[0].distance;
        }
        if (cell->connections[0].distance + distanceDelta < minDistance) {
            distanceDelta = minDistance - cell->connections[0].distance;
        }
        cell->connections[0].distance += distanceDelta;
        auto& connectedCell = cell->connections[0].cell;
        connectedCell->getRefDistance(cell) += distanceDelta;

        // Apply impulse
        if (!crawling.impulseAlreadyApplied) {
            if ((distanceDelta < 0 && cell->connections[0].distance < crawling.initialDistance)
                || (distanceDelta > 0 && cell->connections[0].distance > crawling.initialDistance)) {
                crawling.impulseAlreadyApplied = true;

                auto power = min(1.0f, abs(distanceDelta));
                if (crawling.forward) {
                    power *= powf(1.0f - crawling.forwardBackwardRatio, 4.0f);
                } else {
                    power *= powf(crawling.forwardBackwardRatio, 4.0f);
                }
                auto direction = calcAverageDirection(data, cell);

                auto front = Math::rotateClockwise(data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos), cell->frontAngle);
                if (Math::dot(front, direction) > 0) {
                    direction *= -1.0f;
                }
                if (!crawling.forward) {
                    direction *= -1.0f;
                }

                auto acceleration = direction * power * cudaSimulationParameters.muscleCrawlingAcceleration.value[cell->color] * 5;
                applyAcceleration(cell, acceleration);
            }
        }

        crawling.lastActualDistance = actualDistance;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
        --crawling.activationCountdown;
    }
}

__inline__ __device__ void MuscleProcessor::manualCrawling(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& muscle = cell->cellTypeData.muscle;
    auto& crawling = muscle.modeData.manualCrawling;

    if (cell->numConnections != 1 && cell->numConnections != 2) {
        return;
    }
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (crawling.initialDistance == VALUE_NOT_SET_FLOAT) {
        crawling.initialDistance = cell->connections[0].distance;
        crawling.lastActualDistance = data.cellMap.getDistance(cell->connections[0].cell->pos, cell->pos);
        crawling.lastDistanceDelta = 0;
        crawling.impulseAlreadyApplied = true;
    }

    // Process manual crawling
    if (SignalProcessor::isManuallyTriggered(data, cell)) {

        auto actualDistance = data.cellMap.getDistance(cell->connections[0].cell->pos, cell->pos);
        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));

        // Calc min and max distance
        auto maxDistanceDeviation = max(0.0f, min(1.0f, crawling.maxDistanceDeviation));
        auto maxDistance = max(crawling.initialDistance * (1.0f + maxDistanceDeviation), MinDistance);
        auto minDistance = min(max(crawling.initialDistance * (1.0f - maxDistanceDeviation), MinDistance), crawling.initialDistance);

        // Calc and apply distance delta
        auto distanceDelta = activation > 0 ? 1.05f - crawling.forwardBackwardRatio : 0.05f + crawling.forwardBackwardRatio;
        distanceDelta *= 0.25f * activation;
        if (cell->connections[0].distance + distanceDelta > maxDistance) {
            distanceDelta = maxDistance - cell->connections[0].distance;
        }
        if (cell->connections[0].distance + distanceDelta < minDistance) {
            distanceDelta = minDistance - cell->connections[0].distance;
        }
        cell->connections[0].distance += distanceDelta;
        auto& connectedCell = cell->connections[0].cell;
        connectedCell->getRefDistance(cell) += distanceDelta;

        if ((distanceDelta > 0 && crawling.lastDistanceDelta <= 0) || (distanceDelta < 0 && crawling.lastDistanceDelta >= 0)) {
            crawling.impulseAlreadyApplied = false;
        }
        crawling.lastDistanceDelta = distanceDelta;

        // Apply impulse
        if (!crawling.impulseAlreadyApplied) {
            if ((distanceDelta < 0 && cell->connections[0].distance < crawling.initialDistance)
                || (distanceDelta > 0 && cell->connections[0].distance > crawling.initialDistance)) {
                crawling.impulseAlreadyApplied = true;

                auto power = min(5.0f, abs(distanceDelta));
                if (activation > 0) {
                    power *= powf(1.0f - crawling.forwardBackwardRatio, 4.0f);
                } else {
                    power *= powf(crawling.forwardBackwardRatio, 4.0f);
                }
                auto direction = calcAverageDirection(data, cell);

                auto front = Math::rotateClockwise(data.cellMap.getCorrectedDirection(cell->connections[0].cell->pos - cell->pos), cell->frontAngle);
                if (Math::dot(front, direction) > 0) {
                    direction *= -1.0f;
                }
                if (activation < 0) {
                    direction *= -1.0f;
                }

                auto acceleration = direction * power * cudaSimulationParameters.muscleCrawlingAcceleration.value[cell->color] * 20;
                applyAcceleration(cell, acceleration);
            }
        }

        crawling.lastActualDistance = actualDistance;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
}

__inline__ __device__ void MuscleProcessor::directMovement(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (cell->frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        auto direction = SignalProcessor::calcReferenceDirection(data, cell);
        auto angle = Math::getNormalizedAngle(cell->frontAngle + max(-1.0f, min(1.0f, cell->signal.channels[Channels::MuscleAngle])) * 180.0f, -180.0f);
        direction = Math::rotateClockwise(direction, angle);

        auto activation = max(-1.0f, min(1.0f, cell->signal.channels[Channels::CellTypeActivation]));
        direction = direction * cudaSimulationParameters.muscleMovementAcceleration.value[cell->color] * activation * 0.005f;
        cell->vel += direction;
        cell->cellTypeData.muscle.lastMovementX = direction.x;
        cell->cellTypeData.muscle.lastMovementY = direction.y;
        statistics.incNumMuscleActivities(cell->color);
        radiate(data, cell);
    }
}

__inline__ __device__ void
MuscleProcessor::restoreInitialAngleFromPreviousIntern(float& initialAngle, Cell* muscleCell, Cell* affectedCell, int connectionIndex)
{
    if (initialAngle != VALUE_NOT_SET_FLOAT) {
        auto& angle = affectedCell->connections[connectionIndex].angleFromPrevious;
        auto& nextAngle = affectedCell->connections[(connectionIndex + 1) % affectedCell->numConnections].angleFromPrevious;
        auto diff = initialAngle - angle;

        // Check for enough angle space
        if (!(diff > 0 && nextAngle <= diff)) {
            angle = initialAngle;
            nextAngle -= diff;
        }
        initialAngle = VALUE_NOT_SET_FLOAT;
    }
}

__inline__ __device__ void MuscleProcessor::radiate(SimulationData& data, Cell* cell)
{
    auto cellTypeMuscleEnergyCost = cudaSimulationParameters.muscleEnergyCost.value[cell->color];
    if (cellTypeMuscleEnergyCost > 0) {
        EnergyParticleProcessor::radiate(data, cell, cellTypeMuscleEnergyCost);
    }
}

__inline__ __device__ void MuscleProcessor::getChain(Cell** chain, int& chainLength, Cell* startCell)
{
    chain[0] = startCell;

    if (startCell->numConnections == 1) {
        chain[1] = startCell->connections[0].cell;
    } else if (startCell->numConnections == 2) {
        chain[1] = startCell->connections[1].cell;
    } else {
        CUDA_CHECK(false);
    }
    chainLength = 2;

    for (int i = 1; i < MaxChainLength - 1; ++i) {
        auto const& currentCell = chain[i];
        if (currentCell->numConnections != 2) {
            break;
        }
        auto foundNextCell = false;
        for (int j = 0; j < currentCell->numConnections; ++j) {
            auto const& connectedCell = currentCell->connections[j].cell;
            if (connectedCell != chain[i - 1]) {
                chain[i + 1] = connectedCell;
                foundNextCell = true;
                break;
            }
        }
        if (!foundNextCell) {
            break;
        }
    }
}

__inline__ __device__ float2 MuscleProcessor::calcAverageDirection(SimulationData& data, Cell* cell)
{
    Cell* chain[MaxChainLength];
    int chainLength;
    getChain(chain, chainLength, cell);

    float2 result{0, 0};
    for (int i = 0; i < chainLength - 1; ++i) {
        result += Math::getNormalized(data.cellMap.getCorrectedDirection(chain[i]->pos - chain[i + 1]->pos));
    }
    result /= toFloat(chainLength);
    return result;
}

__inline__ __device__ void MuscleProcessor::applyAcceleration(Cell* cell, float2 const& acceleration)
{
    Cell* chain[MaxChainLength];
    int chainLength;
    getChain(chain, chainLength, cell);
    float2 accPerCell{
        min(AccelerationLimit, max(-AccelerationLimit, acceleration.x / chainLength)),
        min(AccelerationLimit, max(-AccelerationLimit, acceleration.y / chainLength))};
    for (int i = 0; i < chainLength; ++i) {
        atomicAdd(&chain[i]->vel.x, accPerCell.x);
        atomicAdd(&chain[i]->vel.y, accPerCell.y);
    }
}

__inline__ __device__ MuscleProcessor::BendingInfo MuscleProcessor::getBendingInfo(Cell* cell)
{
    BendingInfo result;
    if (cell->numConnections == 2) {
        auto privotCell = cell->connections[1].cell;
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

__inline__ __device__ bool MuscleProcessor::isLeftSide(Cell* cell)
{
    if (cell->numConnections == 2) {
        return cell->frontAngle > NEAR_ZERO;
    } else {
        return cell->frontAngle < -NEAR_ZERO;
    }
}
