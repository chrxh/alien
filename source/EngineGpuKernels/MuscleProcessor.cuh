#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "Entities.cuh"
#include "NeuronProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class MuscleProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

    __inline__ __device__ static float getInitialAngleFromPrevious(Object* object, int connectionIndex);  // Return the angleFromPrevious without muscle distortions
    __inline__ __device__ static void restoreInitialAngleFromPrevious(Object* muscleCell, Object* affectedCell, int connectionIndex);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void autoBending(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void manualBending(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void angleBending(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void autoCrawling(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void manualCrawling(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void directMovement(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void restoreInitialAngleFromPreviousIntern(float& initialAngle, Object* muscleCell, Object* affectedCell, int connectionIndex);

    __inline__ __device__ static void radiate(SimulationData& data, Object* object);

    __inline__ __device__ static void getChain(Object** chain, int& chainLength, Object* startCell);
    __inline__ __device__ static float2 calcAverageDirection(SimulationData& data, Object* object);
    __inline__ __device__ static void applyAcceleration(Object* object, float2 const& acceleration);

    struct BendingInfo
    {
        Object* pivotCell;
        ObjectConnection* connection;
        ObjectConnection* connectionPrev;
        ObjectConnection* connectionNext;
    };
    __inline__ __device__ static BendingInfo getBendingInfo(Object* object);

    __inline__ __device__ static bool isLeftSide(Object* object);

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
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, statistics, operations.at(i).object);
    }
}

__device__ __inline__ float MuscleProcessor::getInitialAngleFromPrevious(Object* object, int connectionIndex)
{
    CUDA_CHECK(connectionIndex < object->numConnections);
    for (int i = 0, j = object->numConnections; i < j; ++i) {
        auto connectedObject = object->connections[i].object;
        if (connectedObject->typeData.cell.cellType == CellType_Muscle
            && (connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending)) {

            auto const& initialAngle = connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending
                ? connectedObject->typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle
                : connectedObject->typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle;
            if (initialAngle != VALUE_NOT_SET_FLOAT) {
                auto bendingInfo = MuscleProcessor::getBendingInfo(connectedObject);
                if (bendingInfo.connection == &object->connections[connectionIndex]) {
                    return initialAngle;
                }
                if (bendingInfo.connectionNext == &object->connections[connectionIndex]) {
                    return bendingInfo.connectionNext->angleFromPrevious - (initialAngle - bendingInfo.connection->angleFromPrevious);
                }
            }
        }
    }
    return object->connections[connectionIndex].angleFromPrevious;
}

__device__ __inline__ void MuscleProcessor::restoreInitialAngleFromPrevious(Object* muscleCell, Object* affectedCell, int connectionIndex)
{
    auto& muscle = muscleCell->typeData.cell.cellTypeData.muscle;
    if (muscle.mode == MuscleMode_AutoBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.autoBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    } else if (muscle.mode == MuscleMode_ManualBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.manualBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    } else if (muscle.mode == MuscleMode_AngleBending) {
        restoreInitialAngleFromPreviousIntern(muscle.modeData.angleBending.initialAngle, muscleCell, affectedCell, connectionIndex);
    }
}

__device__ __inline__ void MuscleProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (object->fixed) {
        return;
    }
    object->typeData.cell.cellTypeData.muscle.lastMovementX = 0;
    object->typeData.cell.cellTypeData.muscle.lastMovementY = 0;

    switch (object->typeData.cell.cellTypeData.muscle.mode % MuscleMode_Count) {
    case MuscleMode_AutoBending: {
        autoBending(data, statistics, object);
    } break;
    case MuscleMode_ManualBending: {
        manualBending(data, statistics, object);
    } break;
    case MuscleMode_AngleBending: {
        angleBending(data, statistics, object);
    } break;
    case MuscleMode_AutoCrawling: {
        autoCrawling(data, statistics, object);
    } break;
    case MuscleMode_ManualCrawling: {
        manualCrawling(data, statistics, object);
    } break;
    case MuscleMode_DirectMovement: {
        directMovement(data, statistics, object);
    } break;
    }
}

__inline__ __device__ void MuscleProcessor::autoBending(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& muscle = object->typeData.cell.cellTypeData.muscle;
    auto& bending = muscle.modeData.autoBending;

    if (object->numConnections != 1 && object->numConnections != 2) {
        return;
    }
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }
    // Activation
    if (bending.activationCountdown == 0) {
        bending.activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));
        auto targetAngle = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::getNormalizedAngle(targetAngle + object->typeData.cell.frontAngle, -180.0f);

        auto angleFactor = [&] {
            if (isLeftSide(object)) {
                targetAngleRelToConnection0 = -targetAngleRelToConnection0;
            }
            if (object->numConnections == 1) {
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

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(object);

        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        auto pivotCell = bendingInfo.pivotCell;
        for (int k = 0, l = pivotCell->numConnections; k < l; ++k) {
            auto connectedObject = pivotCell->connections[k].object;
            if (connectedObject->typeData.cell.cellType == CellType_Muscle
                && (connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending)) {
                auto const& initialAngle = connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending
                    ? connectedObject->typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle
                    : connectedObject->typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle;
                if (initialAngle != VALUE_NOT_SET_FLOAT) {
                    auto connectedBendingInfo = MuscleProcessor::getBendingInfo(connectedObject);
                    if (connectedBendingInfo.connection == bendingInfo.connection) {
                        bending.initialAngle = initialAngle;
                        break;
                    }
                }
            }
        }

        bending.forward = !isLeftSide(object);
        bending.impulseAlreadyApplied = true;
    }

    // Process auto bending
    if (NeuronProcessor::isAutoTriggered(data, object, AutoTriggerInterval)) {

        auto forwardBackwardRatio = isLeftSide(object) ? bending.forwardBackwardRatio : 1.0f - bending.forwardBackwardRatio;

        auto bendingInfo = getBendingInfo(object);
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

                auto direction = calcAverageDirection(data, object);

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
                auto acceleration = direction * angleDelta * cudaSimulationParameters.muscleBendingAcceleration.value[object->color] / 9.0f;
                applyAcceleration(object, acceleration);
            }
        }

        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
        --bending.activationCountdown;
    }
}

__inline__ __device__ void MuscleProcessor::manualBending(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& muscle = object->typeData.cell.cellTypeData.muscle;
    auto& bending = muscle.modeData.manualBending;

    if (object->numConnections != 1 && object->numConnections != 2) {
        return;
    }
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(object);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
        auto pivotCell = bendingInfo.pivotCell;
        for (int k = 0, l = pivotCell->numConnections; k < l; ++k) {
            auto connectedObject = pivotCell->connections[k].object;
            if (connectedObject->typeData.cell.cellType == CellType_Muscle
                && (connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending || connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_ManualBending)) {
                auto const& initialAngle = connectedObject->typeData.cell.cellTypeData.muscle.mode == MuscleMode_AutoBending
                    ? connectedObject->typeData.cell.cellTypeData.muscle.modeData.autoBending.initialAngle
                    : connectedObject->typeData.cell.cellTypeData.muscle.modeData.manualBending.initialAngle;
                if (initialAngle != VALUE_NOT_SET_FLOAT) {
                    auto connectedBendingInfo = MuscleProcessor::getBendingInfo(connectedObject);
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
    if (NeuronProcessor::isManuallyTriggered(data, object)) {

        auto bendingInfo = getBendingInfo(object);
        auto activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));

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
        if (isLeftSide(object)) {
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

                auto direction = calcAverageDirection(data, object);

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
                auto acceleration = direction * angleDelta * cudaSimulationParameters.muscleBendingAcceleration.value[object->color] / 9.0f;
                applyAcceleration(object, acceleration);
            }
        }

        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
    }
}

__inline__ __device__ void MuscleProcessor::angleBending(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& muscle = object->typeData.cell.cellTypeData.muscle;
    auto& bending = muscle.modeData.angleBending;

    if (object->numConnections != 1 && object->numConnections != 2) {
        return;
    }
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (bending.initialAngle == VALUE_NOT_SET_FLOAT) {
        auto bendingInfo = getBendingInfo(object);
        bending.initialAngle = bendingInfo.connection->angleFromPrevious;
    }

    // Process angle bending
    if (NeuronProcessor::isManuallyTriggered(data, object)) {

        auto bendingInfo = getBendingInfo(object);
        auto activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));
        auto targetAngle = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::MuscleAngle])) * 180.f;
        auto targetAngleRelToConnection0 = Math::getNormalizedAngle(object->typeData.cell.frontAngle + targetAngle, -180.0f);

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
        if (object->numConnections == 1) {
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
        object->typeData.cell.frontAngle -= angleDelta;

        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
    }
}

__inline__ __device__ void MuscleProcessor::autoCrawling(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& muscle = object->typeData.cell.cellTypeData.muscle;
    auto& crawling = muscle.modeData.autoCrawling;

    if (object->numConnections != 1 && object->numConnections != 2) {
        return;
    }
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Activation
    crawling.activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));
    crawling.activationCountdown = cudaSimulationParameters.muscleActivationCountdown;
    if (crawling.activationCountdown == 0) {
        return;
    }

    // Initialization
    if (crawling.initialDistance == VALUE_NOT_SET_FLOAT) {
        crawling.initialDistance = object->connections[0].distance;
        crawling.forward = true;
        crawling.lastActualDistance = data.objectMap.getDistance(object->connections[0].object->pos, object->pos);
        crawling.impulseAlreadyApplied = true;
    }

    // Process auto crawling
    if (NeuronProcessor::isAutoTriggered(data, object, AutoTriggerInterval)) {

        auto actualDistance = data.objectMap.getDistance(object->connections[0].object->pos, object->pos);
        auto activation = crawling.activation * toFloat(crawling.activationCountdown) / cudaSimulationParameters.muscleActivationCountdown;

        // Change crawling direction
        auto maxDistanceDeviation = max(0.0f, min(1.0f, crawling.maxDistanceDeviation));
        auto maxDistance = max(crawling.initialDistance * (1.0f + maxDistanceDeviation), MinDistance);
        auto minDistance = min(max(crawling.initialDistance * (1.0f - maxDistanceDeviation), MinDistance), crawling.initialDistance);

        if (object->connections[0].distance > maxDistance - NEAR_ZERO) {
            crawling.forward = activation >= 0;
            crawling.impulseAlreadyApplied = false;
        }
        if (object->connections[0].distance < minDistance + NEAR_ZERO) {
            crawling.forward = activation < 0;
            crawling.impulseAlreadyApplied = false;
        }

        // Calc and apply distance delta
        auto distanceDelta = crawling.forward ? -1.05f + crawling.forwardBackwardRatio : 0.05f + crawling.forwardBackwardRatio;
        distanceDelta *= 0.25f * activation;
        if (object->connections[0].distance + distanceDelta > maxDistance) {
            distanceDelta = maxDistance - object->connections[0].distance;
        }
        if (object->connections[0].distance + distanceDelta < minDistance) {
            distanceDelta = minDistance - object->connections[0].distance;
        }
        object->connections[0].distance += distanceDelta;
        auto& connectedObject = object->connections[0].object;
        connectedObject->getRefDistance(object) += distanceDelta;

        // Apply impulse
        if (!crawling.impulseAlreadyApplied) {
            if ((distanceDelta < 0 && object->connections[0].distance < crawling.initialDistance)
                || (distanceDelta > 0 && object->connections[0].distance > crawling.initialDistance)) {
                crawling.impulseAlreadyApplied = true;

                auto power = min(1.0f, abs(distanceDelta));
                if (crawling.forward) {
                    power *= powf(1.0f - crawling.forwardBackwardRatio, 4.0f);
                } else {
                    power *= powf(crawling.forwardBackwardRatio, 4.0f);
                }
                auto direction = calcAverageDirection(data, object);

                auto front = Math::rotateClockwise(data.objectMap.getCorrectedDirection(object->connections[0].object->pos - object->pos), object->typeData.cell.frontAngle);
                if (Math::dot(front, direction) > 0) {
                    direction *= -1.0f;
                }
                if (!crawling.forward) {
                    direction *= -1.0f;
                }

                auto acceleration = direction * power * cudaSimulationParameters.muscleCrawlingAcceleration.value[object->color] * 5;
                applyAcceleration(object, acceleration);
            }
        }

        crawling.lastActualDistance = actualDistance;
        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
        --crawling.activationCountdown;
    }
}

__inline__ __device__ void MuscleProcessor::manualCrawling(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& muscle = object->typeData.cell.cellTypeData.muscle;
    auto& crawling = muscle.modeData.manualCrawling;

    if (object->numConnections != 1 && object->numConnections != 2) {
        return;
    }
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    // Initialization
    if (crawling.initialDistance == VALUE_NOT_SET_FLOAT) {
        crawling.initialDistance = object->connections[0].distance;
        crawling.lastActualDistance = data.objectMap.getDistance(object->connections[0].object->pos, object->pos);
        crawling.lastDistanceDelta = 0;
        crawling.impulseAlreadyApplied = true;
    }

    // Process manual crawling
    if (NeuronProcessor::isManuallyTriggered(data, object)) {

        auto actualDistance = data.objectMap.getDistance(object->connections[0].object->pos, object->pos);
        auto activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));

        // Calc min and max distance
        auto maxDistanceDeviation = max(0.0f, min(1.0f, crawling.maxDistanceDeviation));
        auto maxDistance = max(crawling.initialDistance * (1.0f + maxDistanceDeviation), MinDistance);
        auto minDistance = min(max(crawling.initialDistance * (1.0f - maxDistanceDeviation), MinDistance), crawling.initialDistance);

        // Calc and apply distance delta
        auto distanceDelta = activation > 0 ? 1.05f - crawling.forwardBackwardRatio : 0.05f + crawling.forwardBackwardRatio;
        distanceDelta *= 0.25f * activation;
        if (object->connections[0].distance + distanceDelta > maxDistance) {
            distanceDelta = maxDistance - object->connections[0].distance;
        }
        if (object->connections[0].distance + distanceDelta < minDistance) {
            distanceDelta = minDistance - object->connections[0].distance;
        }
        object->connections[0].distance += distanceDelta;
        auto& connectedObject = object->connections[0].object;
        connectedObject->getRefDistance(object) += distanceDelta;

        if ((distanceDelta > 0 && crawling.lastDistanceDelta <= 0) || (distanceDelta < 0 && crawling.lastDistanceDelta >= 0)) {
            crawling.impulseAlreadyApplied = false;
        }
        crawling.lastDistanceDelta = distanceDelta;

        // Apply impulse
        if (!crawling.impulseAlreadyApplied) {
            if ((distanceDelta < 0 && object->connections[0].distance < crawling.initialDistance)
                || (distanceDelta > 0 && object->connections[0].distance > crawling.initialDistance)) {
                crawling.impulseAlreadyApplied = true;

                auto power = min(5.0f, abs(distanceDelta));
                if (activation > 0) {
                    power *= powf(1.0f - crawling.forwardBackwardRatio, 4.0f);
                } else {
                    power *= powf(crawling.forwardBackwardRatio, 4.0f);
                }
                auto direction = calcAverageDirection(data, object);

                auto front = Math::rotateClockwise(data.objectMap.getCorrectedDirection(object->connections[0].object->pos - object->pos), object->typeData.cell.frontAngle);
                if (Math::dot(front, direction) > 0) {
                    direction *= -1.0f;
                }
                if (activation < 0) {
                    direction *= -1.0f;
                }

                auto acceleration = direction * power * cudaSimulationParameters.muscleCrawlingAcceleration.value[object->color] * 20;
                applyAcceleration(object, acceleration);
            }
        }

        crawling.lastActualDistance = actualDistance;
        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
    }
}

__inline__ __device__ void MuscleProcessor::directMovement(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (object->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }
    if (NeuronProcessor::isManuallyTriggered(data, object)) {
        auto direction = ObjectConnectionProcessor::calcReferenceDirection(data, object);
        auto angle = Math::getNormalizedAngle(object->typeData.cell.frontAngle + max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::MuscleAngle])) * 180.0f, -180.0f);
        direction = Math::rotateClockwise(direction, angle);

        auto activation = max(-1.0f, min(1.0f, object->typeData.cell.signal.channels[Channels::CellTypeActivation]));
        direction = direction * cudaSimulationParameters.muscleMovementAcceleration.value[object->color] * activation * 0.005f;
        object->vel += direction;
        object->typeData.cell.cellTypeData.muscle.lastMovementX = direction.x;
        object->typeData.cell.cellTypeData.muscle.lastMovementY = direction.y;
        statistics.incNumMuscleActivities(object->color);
        radiate(data, object);
    }
}

__inline__ __device__ void
MuscleProcessor::restoreInitialAngleFromPreviousIntern(float& initialAngle, Object* muscleCell, Object* affectedCell, int connectionIndex)
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

__inline__ __device__ void MuscleProcessor::radiate(SimulationData& data, Object* object)
{
    auto cellTypeMuscleEnergyCost = cudaSimulationParameters.muscleEnergyCost.value[object->color];
    if (cellTypeMuscleEnergyCost > 0) {
        EnergyProcessor::radiate(data, object, cellTypeMuscleEnergyCost);
    }
}

__inline__ __device__ void MuscleProcessor::getChain(Object** chain, int& chainLength, Object* startCell)
{
    chain[0] = startCell;

    if (startCell->numConnections == 1) {
        chain[1] = startCell->connections[0].object;
    } else if (startCell->numConnections == 2) {
        chain[1] = startCell->connections[1].object;
    } else {
        CUDA_CHECK(false);
    }
    chainLength = 2;

    for (int i = 1; i < MaxChainLength - 1; ++i) {
        auto const& currentObject = chain[i];
        if (currentObject->numConnections != 2) {
            break;
        }
        auto foundNextCell = false;
        for (int j = 0; j < currentObject->numConnections; ++j) {
            auto const& connectedObject = currentObject->connections[j].object;
            if (connectedObject != chain[i - 1]) {
                chain[i + 1] = connectedObject;
                foundNextCell = true;
                break;
            }
        }
        if (!foundNextCell) {
            break;
        }
    }
}

__inline__ __device__ float2 MuscleProcessor::calcAverageDirection(SimulationData& data, Object* object)
{
    Object* chain[MaxChainLength];
    int chainLength;
    getChain(chain, chainLength, object);

    float2 result{0, 0};
    for (int i = 0; i < chainLength - 1; ++i) {
        result += Math::getNormalized(data.objectMap.getCorrectedDirection(chain[i]->pos - chain[i + 1]->pos));
    }
    result /= toFloat(chainLength);
    return result;
}

__inline__ __device__ void MuscleProcessor::applyAcceleration(Object* object, float2 const& acceleration)
{
    Object* chain[MaxChainLength];
    int chainLength;
    getChain(chain, chainLength, object);
    float2 accPerCell{
        min(AccelerationLimit, max(-AccelerationLimit, acceleration.x / chainLength)),
        min(AccelerationLimit, max(-AccelerationLimit, acceleration.y / chainLength))};
    for (int i = 0; i < chainLength; ++i) {
        atomicAdd(&chain[i]->vel.x, accPerCell.x);
        atomicAdd(&chain[i]->vel.y, accPerCell.y);
    }
}

__inline__ __device__ MuscleProcessor::BendingInfo MuscleProcessor::getBendingInfo(Object* object)
{
    BendingInfo result;
    if (object->numConnections == 2) {
        auto privotCell = object->connections[1].object;
        result.pivotCell = privotCell;
        for (int i = 0; i < privotCell->numConnections; ++i) {
            if (privotCell->connections[i].object == object) {
                result.connection = &privotCell->connections[i];
                result.connectionPrev = &privotCell->connections[(i + privotCell->numConnections - 1) % privotCell->numConnections];
                result.connectionNext = &privotCell->connections[(i + 1) % privotCell->numConnections];
                break;
            }
        }
    }

    // numConnections == 1
    else {
        auto privotCell = object->connections[0].object;
        result.pivotCell = privotCell;
        for (int i = 0; i < privotCell->numConnections; ++i) {
            if (privotCell->connections[i].object == object) {
                result.connection = &privotCell->connections[i];
                result.connectionPrev = &privotCell->connections[(i + privotCell->numConnections - 1) % privotCell->numConnections];
                result.connectionNext = &privotCell->connections[(i + 1) % privotCell->numConnections];
                break;
            }
        }
    }
    return result;
}

__inline__ __device__ bool MuscleProcessor::isLeftSide(Object* object)
{
    if (object->numConnections == 2) {
        return object->typeData.cell.frontAngle > NEAR_ZERO;
    } else {
        return object->typeData.cell.frontAngle < -NEAR_ZERO;
    }
}
