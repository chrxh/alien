#pragma once

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "Definitions.cuh"
#include "EntityFactory.cuh"
#include "ParameterCalculator.cuh"
#include "EnergyProcessor.cuh"
#include "SimulationData.cuh"

class ObjectConnectionProcessor
{
public:
    __inline__ __device__ static void scheduleAddConnectionPair(SimulationData& data, Object* object1, Object* object2);
    __inline__ __device__ static void scheduleDeleteAllConnections(SimulationData& data, Object* object);
    __inline__ __device__ static void scheduleDeleteConnectionPair(SimulationData& data, Object* object1, Object* object2);
    __inline__ __device__ static void scheduleDeleteObject(SimulationData& data, uint64_t const& objectIndex);

    __inline__ __device__ static void processAddOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteObjectOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteConnectionOperations(SimulationData& data);

    // angle of object1 is given by desiredRelAngle with respect to the inserted connection and between [0, +360)
    // angle of object2 will be automatically determined by current geometry
    // if desiredRelAngle=0: angle of object1 will be automatically determined by current geometry
    // if desiredDistance=0: distance will be automatically determined by current geometry
    __inline__ __device__ static bool tryAddConnectionWithRelativeAngle(SimulationData& data, Object* object1, Object* object2, float desiredDistance = 0, float desiredRelAngle = 0);

    // angle of object1 is given by desiredRelAngle with respect to connections[0] and between [0, +360)
    // angle of object2 will be automatically determined by current geometry
    __inline__ __device__ static bool
    tryAddConnectionWithAbsAngle(SimulationData& data, Object* object1, Object* object2, float desiredDistance, float desiredAbsAngle);

    __inline__ __device__ static void deleteConnections(Object* object1, Object* object2);
    __inline__ __device__ static void deleteConnectionOneWay(Object* object1, Object* object2);

    __inline__ __device__ static bool existCrossingConnections(SimulationData& data, float2 const& pos1, float2 const& pos2, float const& radius, bool detached);
    __inline__ __device__ static bool checkConnectedObjectsForCrossingConnection(Object* object1, float2 otherObjectPos);
    __inline__ __device__ static bool hasAngleSpace(SimulationData& data, Object* object, float angle, ConstructorAngleAlignment angleAlignment);
    __inline__ __device__ static bool isConnectedConnected(Object* object, Object* otherObject);

    struct ReferenceAndActualAngle
    {
        float referenceAngle;
        float actualAngle;
    };
    __inline__ __device__ static ReferenceAndActualAngle calcLargestGapReferenceAndActualAngle(SimulationData& data, Object* object, float angleDeviation);

    __inline__ __device__ static bool existsOwnIntersectingObjectInBetween(SimulationData& data, Object* object, Object* otherObject);

    __inline__ __device__ static float2 calcReferenceDirection(SimulationData& data, Object* object);

private:
    static int constexpr MaxOperationsPerCell = 30;

    __inline__ __device__ static void scheduleOperationOnCell(SimulationData& data, Object* object, int operationIndex);

    __inline__ __device__ static void lockAndTryAddConnections(SimulationData& data, Object* object1, Object* object2);

    // angle of object1 is given by desiredRelAngle with respect to the inserted connection and between [0, +360)
    __inline__ __device__ static bool
    tryAddConnectionWithRelAngle_oneWay(
        SimulationData& data,
        Object* object1,
        Object* object2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredRelAngle = 0);

    // angle of object1 is given by desiredRelAngle with respect to connections[0] and between [0, +360)
    __inline__ __device__ static bool
    tryAddConnectionWithAbsAngle_oneWay(Object* object1, Object* object2, float desiredDistance, float desiredAbsAngle);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ObjectConnectionProcessor::scheduleAddConnectionPair(SimulationData& data, Object* object1, Object* object2)
{
    if (object1->type == ObjectType_Fluid || object2->type == ObjectType_Fluid) {
        return;
    }
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::AddConnectionPair;
    operation.data.addConnection.object = object1;
    operation.data.addConnection.otherObject = object2;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void ObjectConnectionProcessor::scheduleDeleteAllConnections(SimulationData& data, Object* object)
{
    auto index = data.structuralOperations.tryGetEntries(object->numConnections * 2);
    if (index == -1) {
        return;
    }

    for (int i = 0; i < object->numConnections; ++i) {
        auto const& connectedObject = object->connections[i].object;
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedObject = object;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, connectedObject, index);
            ++index;
        }
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedObject = connectedObject;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, object, index);
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

__inline__ __device__ void ObjectConnectionProcessor::scheduleDeleteObject(SimulationData& data, uint64_t const& objectIndex)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelObject;
    operation.data.delObject.objectIndex = objectIndex;
    if (data.structuralOperations.tryAddEntry(operation) != -1) {
        auto object = data.entities.objects.at(objectIndex);
        if (object->type == ObjectType_Cell) {
            atomicSub(&object->typeData.cell.creature->numCells, 1u);
        }
    }
}

__inline__ __device__ void ObjectConnectionProcessor::processAddOperations(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::AddConnectionPair == operation.type) {
            lockAndTryAddConnections(data, operation.data.addConnection.object, operation.data.addConnection.otherObject);
        }
    }
}

__inline__ __device__ void ObjectConnectionProcessor::processDeleteObjectOperations(SimulationData& data)
{
    auto partition = calcSystemThreadPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::DelObject == operation.type) {
            auto objectIndex = operation.data.delObject.objectIndex;

            Object* empty = nullptr;
            auto origObject = alienAtomicExch(&data.entities.objects.at(objectIndex), empty);
            if (origObject) {
                EnergyProcessor::createEnergyParticle(data, origObject->pos, origObject->vel, origObject->color, origObject->getEnergy());

                for (int i = 0; i < origObject->numConnections; ++i) {
                    StructuralOperation operation;
                    operation.type = StructuralOperation::Type::DelConnection;
                    operation.data.delConnection.connectedObject = origObject;
                    operation.nextOperationIndex = -1;
                    auto operationIndex = data.structuralOperations.tryAddEntry(operation);
                    if (operationIndex != -1) {
                        scheduleOperationOnCell(data, origObject->connections[i].object, operationIndex);
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
        if (!object) {
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
                    deleteConnectionOneWay(object, operation.data.delConnection.connectedObject);
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

__inline__ __device__ bool ObjectConnectionProcessor::tryAddConnectionWithRelativeAngle(
    SimulationData& data,
    Object* object1,
    Object* object2,
    float desiredDistance,
    float desiredRelAngle)
{
    auto posDelta = object2->pos - object1->pos;
    data.objectMap.correctDirection(posDelta);

    ObjectConnection origConnections[MAX_OBJECT_CONNECTIONS];
    int origNumConnection = object1->numConnections;
    for (int i = 0; i < origNumConnection; ++i) {
        origConnections[i] = object1->connections[i];
    }

    if (!tryAddConnectionWithRelAngle_oneWay(data, object1, object2, posDelta, desiredDistance, desiredRelAngle)) {
        return false;
    }
    if (!tryAddConnectionWithRelAngle_oneWay(data, object2, object1, posDelta * (-1), desiredDistance, 0)) {
        object1->numConnections = origNumConnection;
        for (int i = 0; i < origNumConnection; ++i) {
            object1->connections[i] = origConnections[i];
        }
        return false;
    }
    return true;
}

__inline__ __device__ bool ObjectConnectionProcessor::tryAddConnectionWithAbsAngle(
    SimulationData& data,
    Object* object1,
    Object* object2,
    float desiredDistance,
    float desiredAbsAngle)
{
    auto posDelta = object2->pos - object1->pos;
    data.objectMap.correctDirection(posDelta);

    ObjectConnection origConnections[MAX_OBJECT_CONNECTIONS];
    int origNumConnection = object1->numConnections;
    for (int i = 0; i < origNumConnection; ++i) {
        origConnections[i] = object1->connections[i];
    }

    if (!tryAddConnectionWithAbsAngle_oneWay(object1, object2, desiredDistance, desiredAbsAngle)) {
        return false;
    }
    if (!tryAddConnectionWithRelAngle_oneWay(data, object2, object1, posDelta * (-1), desiredDistance)) {
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

__inline__ __device__ void ObjectConnectionProcessor::lockAndTryAddConnections(SimulationData& data, Object* object1, Object* object2)
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

        if (!alreadyConnected && object1->numConnections < MAX_OBJECT_CONNECTIONS && object2->numConnections < MAX_OBJECT_CONNECTIONS) {

            tryAddConnectionWithRelativeAngle(data, object1, object2);
        }

        lock.releaseLock();
    }
}

__inline__ __device__ bool ObjectConnectionProcessor::tryAddConnectionWithRelAngle_oneWay(
    SimulationData& data,
    Object* object1,
    Object* object2,
    float2 const& posDelta,
    float desiredDistance,
    float desiredRelAngle)
{
    if (object1->numConnections == MAX_OBJECT_CONNECTIONS) {
        return false;
    }

    // !!! Performance intensive !!!
    //if (checkConnectedObjectsForCrossingConnection(object1, object2->pos)) {
    //    return false;
    //}

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
        auto connectedObjectDelta = object1->connections[0].object->pos - object1->pos;
        data.objectMap.correctDirection(connectedObjectDelta);
        auto prevAngle = Math::angleOfVector(connectedObjectDelta);
        auto angleDiff = Math::subtractAngle(newAngle, prevAngle);
        if (0 != desiredRelAngle) {
            angleDiff = desiredRelAngle;
        }
        if (angleDiff < 0) {
            angleDiff += 360.0f;
        }
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
        prevAngle = Math::angleOfVector(data.objectMap.getCorrectedDirection(object1->connections[prevIndex].object->pos - object1->pos));
        nextAngle = Math::angleOfVector(data.objectMap.getCorrectedDirection(object1->connections[index].object->pos - object1->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, newAngle) || prevIndex == index) {
            break;
        }
    }

    // Create new connection object
    ObjectConnection newConnection;
    newConnection.object = object2;
    newConnection.distance = desiredDistance;

    float angleFromPrevious;
    auto refAngle = object1->connections[index].angleFromPrevious;
    if (0 == desiredRelAngle) {
        auto angleDiff1 = Math::subtractAngle(newAngle, prevAngle);
        auto angleDiff2 = Math::subtractAngle(nextAngle, prevAngle);
        auto newAngleFraction = angleDiff2 != 0 ? angleDiff1 / angleDiff2 : 0.5f;
        angleFromPrevious = refAngle * newAngleFraction;
    } else {
        angleFromPrevious = desiredRelAngle;
    }
    angleFromPrevious = min(angleFromPrevious, refAngle);

    if (angleFromPrevious < NEAR_ZERO) {
        return false;
    }
    newConnection.angleFromPrevious = angleFromPrevious;

    // Insert new connection
    if (index == 0) {
        index = object1->numConnections;  // connection at index 0 should be an invariant
    }
    for (int j = object1->numConnections; j > index; --j) {
        object1->connections[j] = object1->connections[j - 1];
    }
    object1->connections[index] = newConnection;
    object1->connections[(index + 1) % (object1->numConnections + 1)].angleFromPrevious = refAngle - angleFromPrevious;
    ++object1->numConnections;

    return true;
}

__inline__ __device__ bool
ObjectConnectionProcessor::tryAddConnectionWithAbsAngle_oneWay(Object* object1, Object* object2, float desiredDistance, float desiredAbsAngle)
{
    if (object1->numConnections == MAX_OBJECT_CONNECTIONS) {
        return false;
    }

    auto const& n = object1->numConnections;
    CUDA_CHECK(n > 0);

    // Find connection index to insert
    auto insertIndex = 0;
    auto summedAngle = 0.0f;
    for (int i = 1; i <= n; ++i) {
        auto const& angleFromPrevious = object1->getConnection(i).angleFromPrevious;

        if (desiredAbsAngle >= summedAngle - NEAR_ZERO && desiredAbsAngle < summedAngle + angleFromPrevious) {
            insertIndex = i;
            break;
        }
        summedAngle += angleFromPrevious;
    }
    CUDA_CHECK(insertIndex > 0);

    // Create new connection object
    ObjectConnection newConnection;
    newConnection.object = object2;
    newConnection.distance = desiredDistance;
    newConnection.angleFromPrevious = desiredAbsAngle - summedAngle;

    // Insert new connection
    auto refAngle = object1->connections[insertIndex].angleFromPrevious;
    for (int j = object1->numConnections; j > insertIndex; --j) {
        object1->connections[j] = object1->connections[j - 1];
    }
    object1->connections[insertIndex] = newConnection;
    object1->connections[(insertIndex + 1) % (object1->numConnections + 1)].angleFromPrevious = refAngle - newConnection.angleFromPrevious;
    ++object1->numConnections;

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
    data.objectMap.executeForEach(pos2, radius, detached, [&](auto const& nearObject) {
        if (!nearObject->tryLock()) {
            return;
        }
        for (int j = 0; j < nearObject->numConnections; ++j) {
            auto const& connectedNearObject = nearObject->connections[j].object;
            if (Math::crossing(pos1, pos2, nearObject->pos, connectedNearObject->pos)) {
                nearObject->releaseLock();
                result = true;
                return;
            }
        }
        nearObject->releaseLock();
    });
    return result;
}

__inline__ __device__ bool ObjectConnectionProcessor::checkConnectedObjectsForCrossingConnection(Object* object1, float2 otherObjectPos)
{
    auto const& n = object1->numConnections;
    if (n < 2) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        auto connectedObject = object1->connections[i].object;
        auto nextConnectedObject = object1->connections[(i + 1) % n].object;
        bool bothConnected = false;
        for (int j = 0; j < connectedObject->numConnections; ++j) {
            if (connectedObject->connections[j].object == nextConnectedObject) {
                bothConnected = true;
                break;
            }
        }
        if (!bothConnected) {
            continue;
        }
        if (Math::crossing(object1->pos, otherObjectPos, connectedObject->pos, nextConnectedObject->pos)) {
            return true;
        }
    }
    return false;
}

__inline__ __device__ bool ObjectConnectionProcessor::hasAngleSpace(SimulationData& data, Object* object, float angle, ConstructorAngleAlignment angleAlignment)
{
    if (angleAlignment == ConstructorAngleAlignment_None) {
        return true;
    }

    int index = 0;
    float prevAngle;
    float nextAngle;
    for (; index < object->numConnections; ++index) {
        auto prevIndex = (index + object->numConnections - 1) % object->numConnections;
        prevAngle = Math::angleOfVector(data.objectMap.getCorrectedDirection(object->connections[prevIndex].object->pos - object->pos));
        nextAngle = Math::angleOfVector(data.objectMap.getCorrectedDirection(object->connections[index].object->pos - object->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, angle) || prevIndex == index) {
            auto const angleUnit = 360.0f / toFloat(angleAlignment + 1);
            return object->connections[index].angleFromPrevious > angleUnit + NEAR_ZERO;
        }
    }
    return true;
}

__inline__ __device__ bool ObjectConnectionProcessor::isConnectedConnected(Object* object, Object* otherObject)
{
    if (object == otherObject) {
        return true;
    }
    bool result = false;
    for (int i = 0; i < otherObject->numConnections; ++i) {
        auto const& connectedObject = otherObject->connections[i].object;
        if (connectedObject == object) {
            result = true;
            break;
        }

        for (int j = 0; j < connectedObject->numConnections; ++j) {
            auto const& connectedConnectedObject = connectedObject->connections[j].object;
            if (connectedConnectedObject == object) {
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
ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(SimulationData& data, Object* object, float angleDeviation)
{
    if (0 == object->numConnections) {
        return ReferenceAndActualAngle{0, data.primaryNumberGen.random() * 360};
    }
    auto displacement = object->connections[0].object->pos - object->pos;
    data.objectMap.correctDirection(displacement);
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

__inline__ __device__ void ObjectConnectionProcessor::scheduleOperationOnCell(SimulationData& data, Object* object, int operationIndex)
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

__inline__ __device__ bool ObjectConnectionProcessor::existsOwnIntersectingObjectInBetween(SimulationData& data, Object* object, Object* otherObject)
{
    auto result = false;
    data.objectMap.executeForEach(object->pos, cudaSimulationParameters.attackerRadius.value[object->color], object->detached, [&](Object* nearObject) {
        if (result) {
            return;
        }
        if (nearObject == object) {
            return;
        }
        if (nearObject == otherObject) {
            return;
        }
        if (object->type == ObjectType_Cell && nearObject->type == ObjectType_Cell && !object->typeData.cell.isSameCreature(&nearObject->typeData.cell)) {
            return;
        }
        for (int i = 0; i < nearObject->numConnections; ++i) {
            auto connectedNearObject = nearObject->connections[i].object;
            if (Math::crossing(nearObject->pos, connectedNearObject->pos, object->pos, otherObject->pos)) {
                result = true;
                return;
            }
        }
    });
    return result;
}

__inline__ __device__ float2 ObjectConnectionProcessor::calcReferenceDirection(SimulationData& data, Object* object)
{
    if (object->numConnections == 0) {
        return float2{0.0f, -1.0f};
    }
    return Math::getNormalized(data.objectMap.getCorrectedDirection(object->connections[0].object->pos - object->pos));
}
