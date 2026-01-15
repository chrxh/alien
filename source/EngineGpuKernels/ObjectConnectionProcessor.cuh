#pragma once

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "Definitions.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "EnergyParticleProcessor.cuh"
#include "SimulationData.cuh"

class ObjectConnectionProcessor
{
public:
    __inline__ __device__ static void scheduleAddConnectionPair(SimulationData& data, Object* object1, Object* object2);
    __inline__ __device__ static void scheduleDeleteAllConnections(SimulationData& data, Object* cell);
    __inline__ __device__ static void scheduleDeleteConnectionPair(SimulationData& data, Object* object1, Object* object2);
    __inline__ __device__ static void scheduleDeleteCell(SimulationData& data, uint64_t const& cellIndex);

    __inline__ __device__ static void processAddOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteCellOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteConnectionOperations(SimulationData& data);

    __inline__ __device__ static bool tryAddConnections(
        SimulationData& data,
        Object* object1,
        Object* object2,
        float desiredAngleOnCell1,
        float desiredAngleOnCell2,
        float desiredDistance,
        ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_None);
    __inline__ __device__ static void deleteConnections(Object* object1, Object* object2);
    __inline__ __device__ static void deleteConnectionOneWay(Object* object1, Object* object2);

    __inline__ __device__ static bool
    existCrossingConnections(SimulationData& data, float2 const& pos1, float2 const& pos2, float const& radius, bool detached);
    __inline__ __device__ static bool checkConnectedCellsForCrossingConnection(Object* object1, float2 otherCellPos);
    __inline__ __device__ static bool hasAngleSpace(SimulationData& data, Object* cell, float angle, ConstructorAngleAlignment angleAlignment);
    __inline__ __device__ static bool isConnectedConnected(Object* cell, Object* otherCell);

    struct ReferenceAndActualAngle
    {
        float referenceAngle;
        float actualAngle;
    };
    __inline__ __device__ static ReferenceAndActualAngle calcLargestGapReferenceAndActualAngle(SimulationData& data, Object* cell, float angleDeviation);

    __inline__ __device__ static bool existsOwnIntersectingCellInBetween(SimulationData& data, Object* cell, Object* otherCell);

private:
    static int constexpr MaxOperationsPerCell = 30;

    __inline__ __device__ static void scheduleOperationOnCell(SimulationData& data, Object* cell, int operationIndex);

    __inline__ __device__ static void lockAndtryAddConnections(SimulationData& data, Object* object1, Object* object2);
    __inline__ __device__ static bool tryAddConnectionOneWay(
        SimulationData& data,
        Object* object1,
        Object* object2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredAngleOnCell1 = 0,
        ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_None);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ObjectConnectionProcessor::scheduleAddConnectionPair(SimulationData& data, Object* object1, Object* object2)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::AddConnectionPair;
    operation.data.addConnection.object = object1;
    operation.data.addConnection.otherObject = object2;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void ObjectConnectionProcessor::scheduleDeleteAllConnections(SimulationData& data, Object* cell)
{
    auto index = data.structuralOperations.tryGetEntries(object->numConnections * 2);
    if (index == -1) {
        return;
    }

    for (int i = 0; i < object->numConnections; ++i) {
        auto const& connectedCell = object->connections[i].object;
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedObject = cell;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, connectedCell, index);
            ++index;
        }
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedObject = connectedCell;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, cell, index);
            ++index;
        }
    }
}

__inline__ __device__ void ObjectConnectionProcessor::scheduleDeleteConnectionPair(SimulationData& data, Object* object1, Object* object2)
{
    StructuralOperation operation1;
    operation1.type = StructuralOperation::Type::DelConnection;
    operation1.data.delConnection.connectedObject = object2;
    operation1.nextOperationIndex = -1;
    auto operationIndex1 = data.structuralOperations.tryAddEntry(operation1);
    if (operationIndex1 != -1) {
        scheduleOperationOnCell(data, object1, operationIndex1);
    } else {
        CUDA_THROW_NOT_IMPLEMENTED();
    }

    StructuralOperation operation2;
    operation2.type = StructuralOperation::Type::DelConnection;
    operation2.data.delConnection.connectedObject = object1;
    operation2.nextOperationIndex = -1;
    auto operationIndex2 = data.structuralOperations.tryAddEntry(operation2);
    if (operationIndex2 != -1) {
        scheduleOperationOnCell(data, object2, operationIndex2);
    } else {
        CUDA_THROW_NOT_IMPLEMENTED();
    }
}

__inline__ __device__ void ObjectConnectionProcessor::scheduleDeleteCell(SimulationData& data, uint64_t const& cellIndex)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelCell;
    operation.data.delCell.cellIndex = cellIndex;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void ObjectConnectionProcessor::processAddOperations(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::AddConnectionPair == operation.type) {
            lockAndtryAddConnections(data, operation.data.addConnection.object, operation.data.addConnection.otherObject);
        }
    }
}

__inline__ __device__ void ObjectConnectionProcessor::processDeleteCellOperations(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::DelCell == operation.type) {
            auto cellIndex = operation.data.delCell.cellIndex;

            Object* empty = nullptr;
            auto origCell = alienAtomicExch(&data.entities.objects.at(cellIndex), empty);
            if (origCell) {
                EnergyParticleProcessor::createEnergyParticle(data, origCell->pos, origCell->vel, origCell->color, origCell->getEnergy());

                for (int i = 0; i < origCell->numConnections; ++i) {
                    StructuralOperation operation;
                    operation.type = StructuralOperation::Type::DelConnection;
                    operation.data.delConnection.connectedObject = origCell;
                    operation.nextOperationIndex = -1;
                    auto operationIndex = data.structuralOperations.tryAddEntry(operation);
                    if (operationIndex != -1) {
                        scheduleOperationOnCell(data, origCell->connections[i].object, operationIndex);
                    } else {
                        CUDA_THROW_NOT_IMPLEMENTED();
                    }
                }
            }
        }
    }
}

__inline__ __device__ void ObjectConnectionProcessor::processDeleteConnectionOperations(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = data.entities.objects.at(index);
        if (!cell) {
            continue;
        }
        auto scheduledOperationIndex = object->scheduledOperationIndex;
        if (scheduledOperationIndex != -1) {
            for (int depth = 0; depth < MaxOperationsPerCell; ++depth) {

                //#TODO check if following `if` can be removed because the condition should never be true
                if (scheduledOperationIndex < 0 || scheduledOperationIndex >= data.structuralOperations.getNumEntries()) {
                    break;
                }
                auto operation = data.structuralOperations.at(scheduledOperationIndex);
                switch (operation.type) {
                case StructuralOperation::Type::DelConnection: {
                    deleteConnectionOneWay(cell, operation.data.delConnection.connectedObject);
                } break;
                default:
                    break;
                }
                scheduledOperationIndex = operation.nextOperationIndex;
                if (scheduledOperationIndex == -1) {
                    break;
                }
            }
            object->scheduledOperationIndex = -1;
        }
    }
}

__inline__ __device__ bool ObjectConnectionProcessor::tryAddConnections(
    SimulationData& data,
    Object* object1,
    Object* object2,
    float desiredAngleOnCell1,
    float desiredAngleOnCell2,
    float desiredDistance,
    ConstructorAngleAlignment angleAlignment)
{
    auto posDelta = object2->pos - object1->pos;
    data.cellMap.correctDirection(posDelta);

    ObjectConnection origConnections[MAX_CELL_BONDS];
    int origNumConnection = object1->numConnections;
    for (int i = 0; i < origNumConnection; ++i) {
        origConnections[i] = object1->connections[i];
    }

    if (!tryAddConnectionOneWay(data, object1, object2, posDelta, desiredDistance, desiredAngleOnCell1, angleAlignment)) {
        return false;
    }
    if (!tryAddConnectionOneWay(data, object2, object1, posDelta * (-1), desiredDistance, desiredAngleOnCell2, angleAlignment)) {
        object1->numConnections = origNumConnection;
        for (int i = 0; i < origNumConnection; ++i) {
            object1->connections[i] = origConnections[i];
        }
        return false;
    }
    return true;
}

__inline__ __device__ void ObjectConnectionProcessor::deleteConnections(Object* object1, Object* object2)
{
    deleteConnectionOneWay(object1, object2);
    deleteConnectionOneWay(object2, object1);
}

__inline__ __device__ void ObjectConnectionProcessor::lockAndtryAddConnections(SimulationData& data, Object* object1, Object* object2)
{
    SystemDoubleLock lock;
    lock.init(&object1->locked, &object2->locked);
    if (lock.tryLock()) {

        bool alreadyConnected = false;
        for (int i = 0; i < object1->numConnections; ++i) {
            if (object1->connections[i].object == object2) {
                alreadyConnected = true;
                break;
            }
        }

        if (!alreadyConnected && object1->numConnections < MAX_CELL_BONDS && object2->numConnections < MAX_CELL_BONDS) {

            tryAddConnections(data, object1, object2, 0, 0, 0);
        }

        lock.releaseLock();
    }
}

__inline__ __device__ bool ObjectConnectionProcessor::tryAddConnectionOneWay(
    SimulationData& data,
    Object* object1,
    Object* object2,
    float2 const& posDelta,
    float desiredDistance,
    float desiredAngleOnCell1,
    ConstructorAngleAlignment angleAlignment)
{
    if (object1->numConnections == MAX_CELL_BONDS) {
        return false;
    }
    if (ConstructorAngleAlignment_None != angleAlignment && object1->numConnections >= angleAlignment + 1) {
        return false;
    }

    // !!! Performance intensive !!!
    //if (checkConnectedCellsForCrossingConnection(object1, object2->pos)) {
    //    return false;
    //}

    angleAlignment %= ConstructorAngleAlignment_Count;

    auto newAngle = Math::angleOfVector(posDelta);
    if (desiredDistance == 0) {
        desiredDistance = Math::length(posDelta);
    }

    // *****
    // special case: object1 has no connections
    // *****
    if (0 == object1->numConnections) {
        object1->numConnections++;
        object1->connections[0].object = object2;
        object1->connections[0].distance = desiredDistance;
        object1->connections[0].angleFromPrevious = 360.0f;
        return true;
    }

    // *****
    // special case: object1 has one connection
    // *****
    if (1 == object1->numConnections) {
        auto connectedCellDelta = object1->connections[0].object->pos - object1->pos;
        data.cellMap.correctDirection(connectedCellDelta);
        auto prevAngle = Math::angleOfVector(connectedCellDelta);
        auto angleDiff = Math::subtractAngle(newAngle, prevAngle);
        if (0 != desiredAngleOnCell1) {
            angleDiff = desiredAngleOnCell1;
        }
        angleDiff = Math::alignAngle(angleDiff, angleAlignment);
        if (angleDiff < 0) {
            angleDiff += 360.0f;
        }
        angleDiff = Math::alignAngleOnBoundaries(angleDiff, 360.0f, angleAlignment);
        if (abs(angleDiff) < NEAR_ZERO || abs(angleDiff - 360.0f) < NEAR_ZERO || abs(angleDiff + 360.0f) < NEAR_ZERO) {
            return false;
        }

        object1->connections[1].angleFromPrevious = angleDiff;
        object1->connections[0].angleFromPrevious = 360.0f - angleDiff;

        object1->numConnections++;
        object1->connections[1].object = object2;
        object1->connections[1].distance = desiredDistance;
        return true;
    }

    // *****
    // process general case
    // *****

    // find appropriate index for new connection
    int index = 0;
    float prevAngle = 0;
    float nextAngle = 0;
    for (; index < object1->numConnections; ++index) {
        auto prevIndex = (index + object1->numConnections - 1) % object1->numConnections;
        prevAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(object1->connections[prevIndex].object->pos - object1->pos));
        nextAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(object1->connections[index].object->pos - object1->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, newAngle) || prevIndex == index) {
            break;
        }
    }

    // create new connection object
    ObjectConnection newConnection;
    newConnection.object = object2;
    newConnection.distance = desiredDistance;

    float angleFromPrevious;
    auto refAngle = object1->connections[index].angleFromPrevious;
    if (0 == desiredAngleOnCell1) {
        auto angleDiff1 = Math::subtractAngle(newAngle, prevAngle);
        auto angleDiff2 = Math::subtractAngle(nextAngle, prevAngle);
        auto newAngleFraction = angleDiff2 != 0 ? angleDiff1 / angleDiff2 : 0.5f;
        angleFromPrevious = refAngle * newAngleFraction;
    } else {
        angleFromPrevious = desiredAngleOnCell1;
    }
    angleFromPrevious = min(angleFromPrevious, refAngle);

    if (angleFromPrevious < NEAR_ZERO) {
        return false;
    }
    newConnection.angleFromPrevious = angleFromPrevious;

    // insert new connection
    if (index == 0) {
        index = object1->numConnections;  // connection at index 0 should be an invariant
    }
    if (index == 0) {
        index = object1->numConnections;  // connection at index 0 should be an invariant
    }
    for (int j = object1->numConnections; j > index; --j) {
        object1->connections[j] = object1->connections[j - 1];
    }
    object1->connections[index] = newConnection;
    object1->connections[(index + 1) % (object1->numConnections + 1)].angleFromPrevious = refAngle - angleFromPrevious;

    // alignment
    if (angleAlignment != ConstructorAngleAlignment_None) {
        auto const angleUnit = 360.0f / toFloat(angleAlignment + 1);

        // align angles
        auto angleAdded = 0.0f;
        for (int i = 0; i < object1->numConnections + 2; ++i) {
            auto index = i % (object1->numConnections + 1);
            object1->connections[index].angleFromPrevious = Math::alignAngle(object1->connections[index].angleFromPrevious, angleAlignment);
            if (angleAdded > NEAR_ZERO && object1->connections[index].angleFromPrevious - angleAdded > NEAR_ZERO) {
                object1->connections[index].angleFromPrevious -= angleUnit;
            }
            if (abs(object1->connections[index].angleFromPrevious) < NEAR_ZERO) {
                object1->connections[index].angleFromPrevious = angleUnit;
                angleAdded = angleUnit;
            } else {
                angleAdded = 0;
            }
        }

        // ensure that sum of angles is 360 deg
        for (int repetition = 0; repetition < MAX_CELL_BONDS; ++repetition) {
            float sumAngle = 0;
            for (int i = 0, j = object1->numConnections + 1; i < j; ++i) {
                sumAngle += object1->connections[i].angleFromPrevious;
            }
            if (sumAngle > 360.0f + NEAR_ZERO || sumAngle < 360.0f - NEAR_ZERO) {
                int indexWithMaxAngle = -1;
                float maxAngle = 0;
                for (int k = 0, l = object1->numConnections + 1; k < l; ++k) {
                    if (object1->connections[k].angleFromPrevious > maxAngle) {
                        maxAngle = object1->connections[k].angleFromPrevious;
                        indexWithMaxAngle = k;
                    }
                }
                if (sumAngle > 360.0f + NEAR_ZERO) {
                    object1->connections[indexWithMaxAngle].angleFromPrevious -= angleUnit;
                } else {
                    object1->connections[indexWithMaxAngle].angleFromPrevious += angleUnit;
                }
            } else {
                break;
            }
        }
    }
    object1->numConnections++;

    return true;
}

__inline__ __device__ void ObjectConnectionProcessor::deleteConnectionOneWay(Object* object1, Object* object2)
{
    for (int i = 0; i < object1->numConnections; ++i) {
        if (object1->connections[i].object == object2) {
            float angleToAdd = object1->connections[i].angleFromPrevious;
            for (int j = i; j < object1->numConnections - 1; ++j) {
                object1->connections[j] = object1->connections[j + 1];
            }

            if (i < object1->numConnections - 1) {
                object1->connections[i].angleFromPrevious += angleToAdd;
            } else {
                object1->connections[0].angleFromPrevious += angleToAdd;
            }

            --object1->numConnections;
            return;
        }
    }
}

__inline__ __device__ bool
ObjectConnectionProcessor::existCrossingConnections(SimulationData& data, float2 const& pos1, float2 const& pos2, float const& radius, bool detached)
{
    auto result = false;
    data.cellMap.executeForEach(pos2, radius, detached, [&](auto const& nearCell) {
        if (!nearCell->tryLock()) {
            return;
        }
        for (int j = 0; j < nearCell->numConnections; ++j) {
            auto const& connectedNearCell = nearCell->connections[j].object;
            if (Math::crossing(pos1, pos2, nearCell->pos, connectedNearCell->pos)) {
                nearCell->releaseLock();
                result = true;
                return;
            }
        }
        nearCell->releaseLock();
    });
    return result;
}

__inline__ __device__ bool ObjectConnectionProcessor::checkConnectedCellsForCrossingConnection(Object* object1, float2 otherCellPos)
{
    auto const& n = object1->numConnections;
    if (n < 2) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        auto connectedCell = object1->connections[i].object;
        auto nextConnectedCell = object1->connections[(i + 1) % n].object;
        bool bothConnected = false;
        for (int j = 0; j < connectedCell->numConnections; ++j) {
            if (connectedCell->connections[j].object == nextConnectedCell) {
                bothConnected = true;
                break;
            }
        }
        if (!bothConnected) {
            continue;
        }
        if (Math::crossing(object1->pos, otherCellPos, connectedCell->pos, nextConnectedCell->pos)) {
            return true;
        }
    }
    return false;
}

__inline__ __device__ bool ObjectConnectionProcessor::hasAngleSpace(SimulationData& data, Object* cell, float angle, ConstructorAngleAlignment angleAlignment)
{
    if (angleAlignment == ConstructorAngleAlignment_None) {
        return true;
    }

    int index = 0;
    float prevAngle;
    float nextAngle;
    for (; index < object->numConnections; ++index) {
        auto prevIndex = (index + object->numConnections - 1) % object->numConnections;
        prevAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(object->connections[prevIndex].object->pos - object->pos));
        nextAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(object->connections[index].object->pos - object->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, angle) || prevIndex == index) {
            auto const angleUnit = 360.0f / toFloat(angleAlignment + 1);
            return object->connections[index].angleFromPrevious > angleUnit + NEAR_ZERO;
        }
    }
    return true;
}

__inline__ __device__ bool ObjectConnectionProcessor::isConnectedConnected(Object* cell, Object* otherCell)
{
    if (cell == otherCell) {
        return true;
    }
    bool result = false;
    for (int i = 0; i < otherCell->numConnections; ++i) {
        auto const& connectedCell = otherCell->connections[i].object;
        if (connectedCell == cell) {
            result = true;
            break;
        }

        for (int j = 0; j < connectedCell->numConnections; ++j) {
            auto const& connectedConnectedCell = connectedCell->connections[j].object;
            if (connectedConnectedCell == cell) {
                result = true;
                break;
            }
        }
        if (result) {
            return true;
        }
    }
    return result;
}

__inline__ __device__ ObjectConnectionProcessor::ReferenceAndActualAngle
ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(SimulationData& data, Object* cell, float angleDeviation)
{
    if (0 == object->numConnections) {
        return ReferenceAndActualAngle{0, data.primaryNumberGen.random() * 360};
    }
    auto displacement = object->connections[0].object->pos - object->pos;
    data.cellMap.correctDirection(displacement);
    auto angle = Math::angleOfVector(displacement);
    int index = 0;
    float largestAngleGap = 0;
    float angleOfLargestAngleGap = 0;
    auto numConnections = object->numConnections;
    for (int i = 1; i <= numConnections; ++i) {
        auto angleDiff = object->connections[i % numConnections].angleFromPrevious;
        if (angleDiff > largestAngleGap) {
            largestAngleGap = angleDiff;
            index = i % numConnections;
            angleOfLargestAngleGap = angle;
        }
        angle += angleDiff;
    }

    auto angleFromPrev = object->connections[index].angleFromPrevious;
    for (int i = 0; i < numConnections - 1; ++i) {
        if (angleDeviation > angleFromPrev / 2) {
            angleDeviation -= angleFromPrev / 2;
            index = (index + 1) % numConnections;
            angleOfLargestAngleGap += angleFromPrev;
            angleFromPrev = object->connections[index].angleFromPrevious;
            angleDeviation = angleDeviation - angleFromPrev / 2;
        }
        if (angleDeviation < -angleFromPrev / 2) {
            angleDeviation += angleFromPrev / 2;
            index = (index + numConnections - 1) % numConnections;
            angleFromPrev = object->connections[index].angleFromPrevious;
            angleDeviation = angleDeviation + angleFromPrev / 2;
            angleOfLargestAngleGap -= angleFromPrev;
        }
    }
    auto angleFromPreviousConnection = angleFromPrev / 2 + angleDeviation;

    if (angleFromPreviousConnection > 360.0f) {
        angleFromPreviousConnection -= 360;
    }
    angleFromPreviousConnection = max(30.0f, min(angleFromPrev - 30.0f, angleFromPreviousConnection));

    return ReferenceAndActualAngle{angleFromPreviousConnection, angleOfLargestAngleGap + angleFromPreviousConnection};
}

__inline__ __device__ void ObjectConnectionProcessor::scheduleOperationOnCell(SimulationData& data, Object* cell, int operationIndex)
{
    auto origOperationIndex = atomicCAS(&object->scheduledOperationIndex, -1, operationIndex);
    for (int depth = 0; depth < MaxOperationsPerCell; ++depth) {
        if (origOperationIndex == -1) {
            break;
        }
        auto& origOperation = data.structuralOperations.at(origOperationIndex);
        origOperationIndex = atomicCAS(&origOperation.nextOperationIndex, -1, operationIndex);
    }
}

__inline__ __device__ bool ObjectConnectionProcessor::existsOwnIntersectingCellInBetween(SimulationData& data, Object* cell, Object* otherCell)
{
    auto result = false;
    data.cellMap.executeForEach(object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](Object* nearCell) {
        if (result) {
            return;
        }
        if (nearCell == cell) {
            return;
        }
        if (nearCell == otherCell) {
            return;
        }
        if (!object->isSameCreature(nearCell)) {
            return;
        }
        for (int i = 0; i < nearCell->numConnections; ++i) {
            auto connectedNearCell = nearCell->connections[i].object;
            if (Math::crossing(nearCell->pos, connectedNearCell->pos, object->pos, otherCell->pos)) {
                result = true;
                return;
            }
        }
    });
    return result;
}
