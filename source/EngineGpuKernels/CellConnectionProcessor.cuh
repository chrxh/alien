#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"
#include "ObjectFactory.cuh"
#include "RadiationProcessor.cuh"

class CellConnectionProcessor
{
public:
    __inline__ __device__ static void scheduleAddConnectionPair(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDeleteAllConnections(SimulationData& data, Cell* cell);
    __inline__ __device__ static void scheduleDeleteConnectionPair(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDeleteCell(SimulationData& data, uint64_t const& cellIndex);

    __inline__ __device__ static void processAddOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteCellOperations(SimulationData& data);
    __inline__ __device__ static void processDeleteConnectionOperations(SimulationData& data);

    __inline__ __device__ static bool tryAddConnections(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float desiredAngleOnCell1,
        float desiredAngleOnCell2,
        float desiredDistance,
        ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_None);
    __inline__ __device__ static void deleteConnections(Cell* cell1, Cell* cell2);
    __inline__ __device__ static void deleteConnectionOneWay(Cell* cell1, Cell* cell2);

    __inline__ __device__ static bool existCrossingConnections(SimulationData& data, float2 pos1, float2 pos2, int detached, int color);
    __inline__ __device__ static bool wouldResultInOverlappingConnection(Cell* cell1, float2 otherCellPos);
    __inline__ __device__ static bool isConnectedConnected(Cell* cell, Cell* otherCell);

    struct ReferenceAndActualAngle
    {
        float referenceAngle;
        float actualAngle;
    };
    __inline__ __device__ static ReferenceAndActualAngle calcLargestGapReferenceAndActualAngle(SimulationData& data, Cell* cell, float angleDeviation);

private:
    static int constexpr MaxOperationsPerCell = 30;

    __inline__ __device__ static void scheduleOperationOnCell(SimulationData& data, Cell* cell, int operationIndex);

    __inline__ __device__ static void lockAndtryAddConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static bool tryAddConnectionOneWay(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredAngleOnCell1 = 0,
        ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_None);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void
CellConnectionProcessor::scheduleAddConnectionPair(SimulationData& data, Cell* cell1, Cell* cell2)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::AddConnectionPair;
    operation.data.addConnection.cell = cell1;
    operation.data.addConnection.otherCell = cell2;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void CellConnectionProcessor::scheduleDeleteAllConnections(SimulationData& data, Cell* cell)
{
    auto index = data.structuralOperations.tryGetEntries(cell->numConnections * 2);
    if (index == -1) {
        return;
    }

    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectedCell = cell->connections[i].cell;
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedCell = cell;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, connectedCell, index);
            ++index;
        }
        {
            StructuralOperation& operation = data.structuralOperations.at(index);
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedCell = connectedCell;
            operation.nextOperationIndex = -1;
            scheduleOperationOnCell(data, cell, index);
            ++index;
        }
    }
}

__inline__ __device__ void
CellConnectionProcessor::scheduleDeleteConnectionPair(SimulationData& data, Cell* cell1, Cell* cell2)
{
    StructuralOperation operation1;
    operation1.type = StructuralOperation::Type::DelConnection;
    operation1.data.delConnection.connectedCell = cell2;
    operation1.nextOperationIndex = -1;
    auto operationIndex1 = data.structuralOperations.tryAddEntry(operation1);
    if (operationIndex1 != -1) {
        scheduleOperationOnCell(data, cell1, operationIndex1);
    } else {
        CUDA_THROW_NOT_IMPLEMENTED();
    }

    StructuralOperation operation2;
    operation2.type = StructuralOperation::Type::DelConnection;
    operation2.data.delConnection.connectedCell = cell1;
    operation2.nextOperationIndex = -1;
    auto operationIndex2 = data.structuralOperations.tryAddEntry(operation2);
    if (operationIndex2 != -1) {
        scheduleOperationOnCell(data, cell2, operationIndex2);
    } else {
        CUDA_THROW_NOT_IMPLEMENTED();
    }
}

__inline__ __device__ void CellConnectionProcessor::scheduleDeleteCell(SimulationData& data, uint64_t const& cellIndex)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelCell;
    operation.data.delCell.cellIndex = cellIndex;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void CellConnectionProcessor::processAddOperations(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::AddConnectionPair == operation.type) {
            lockAndtryAddConnections(
                data,
                operation.data.addConnection.cell,
                operation.data.addConnection.otherCell);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::processDeleteCellOperations(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::DelCell == operation.type) {
            auto cellIndex = operation.data.delCell.cellIndex;

            Cell* empty = nullptr;
            auto origCell = alienAtomicExch(&data.objects.cellPointers.at(cellIndex), empty);
            if (origCell) {
                RadiationProcessor::createEnergyParticle(data, origCell->pos, origCell->vel, origCell->color, origCell->energy);

                for (int i = 0; i < origCell->numConnections; ++i) {
                    StructuralOperation operation;
                    operation.type = StructuralOperation::Type::DelConnection;
                    operation.data.delConnection.connectedCell = origCell;
                    operation.nextOperationIndex = -1;
                    auto operationIndex = data.structuralOperations.tryAddEntry(operation);
                    if (operationIndex != -1) {
                        scheduleOperationOnCell(data, origCell->connections[i].cell, operationIndex);
                    } else {
                        CUDA_THROW_NOT_IMPLEMENTED();
                    }
                }
            }
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::processDeleteConnectionOperations(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.objects.cellPointers.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (!cell) {
            continue;
        }
        auto scheduledOperationIndex = cell->scheduledOperationIndex;
        if (scheduledOperationIndex != -1) {
            for (int depth = 0; depth < MaxOperationsPerCell; ++depth) {

                //#TODO check if following `if` can be removed because the condition should never be true
                if (scheduledOperationIndex < 0 || scheduledOperationIndex >= data.structuralOperations.getNumEntries()) {
                    break;
                }
                auto operation = data.structuralOperations.at(scheduledOperationIndex);
                switch (operation.type) {
                case StructuralOperation::Type::DelConnection: {
                    deleteConnectionOneWay(cell, operation.data.delConnection.connectedCell);
                } break;
                default:
                    break;
                }
                scheduledOperationIndex = operation.nextOperationIndex;
                if (scheduledOperationIndex == -1) {
                    break;
                }
            }
            cell->scheduledOperationIndex = -1;
        }
    }
}

__inline__ __device__ bool CellConnectionProcessor::tryAddConnections(
    SimulationData& data,
    Cell* cell1,
    Cell* cell2,
    float desiredAngleOnCell1,
    float desiredAngleOnCell2,
    float desiredDistance,
    ConstructorAngleAlignment angleAlignment)
{
    auto posDelta = cell2->pos - cell1->pos;
    data.cellMap.correctDirection(posDelta);

    CellConnection origConnections[MAX_CELL_BONDS];
    int origNumConnection = cell1->numConnections;
    for (int i = 0; i < origNumConnection; ++i) {
        origConnections[i] = cell1->connections[i];
    }

    if (!tryAddConnectionOneWay(data, cell1, cell2, posDelta, desiredDistance, desiredAngleOnCell1, angleAlignment)) {
        return false;
    }
    if (!tryAddConnectionOneWay(data, cell2, cell1, posDelta * (-1), desiredDistance, desiredAngleOnCell2, angleAlignment)) {
        cell1->numConnections = origNumConnection;
        for (int i = 0; i < origNumConnection; ++i) {
            cell1->connections[i] = origConnections[i];
        }
        return false;
    }
    return true;
}

__inline__ __device__ void
CellConnectionProcessor::deleteConnections(Cell* cell1, Cell* cell2)
{
    deleteConnectionOneWay(cell1, cell2);
    deleteConnectionOneWay(cell2, cell1);
}

__inline__ __device__ void
CellConnectionProcessor::lockAndtryAddConnections(SimulationData& data, Cell* cell1, Cell* cell2)
{
    SystemDoubleLock lock;
    lock.init(&cell1->locked, &cell2->locked);
    if (lock.tryLock()) {

        bool alreadyConnected = false;
        for (int i = 0; i < cell1->numConnections; ++i) {
            if (cell1->connections[i].cell == cell2) {
                alreadyConnected = true;
                break;
            }
        }

        if (!alreadyConnected && cell1->numConnections < cell1->maxConnections
            && cell2->numConnections < cell2->maxConnections) {

            tryAddConnections(data, cell1, cell2, 0, 0, 0);
        }

        lock.releaseLock();
    }
}

__inline__ __device__ bool CellConnectionProcessor::tryAddConnectionOneWay(
    SimulationData& data,
    Cell* cell1,
    Cell* cell2,
    float2 const& posDelta,
    float desiredDistance,
    float desiredAngleOnCell1,
    ConstructorAngleAlignment angleAlignment)
{
    if (cell1->numConnections == MAX_CELL_BONDS) {
        return false;
    }
    if (ConstructorAngleAlignment_None != angleAlignment && cell1->numConnections >= angleAlignment + 1) {
        return false;
    }

    if (wouldResultInOverlappingConnection(cell1, cell2->pos)) {
        return false;
    }

    angleAlignment %= ConstructorAngleAlignment_Count;

    auto newAngle = Math::angleOfVector(posDelta);
    if (desiredDistance == 0) {
        desiredDistance = Math::length(posDelta);
    }

    // *****
    // special case: cell1 has no connections
    // *****
    if (0 == cell1->numConnections) {
        cell1->numConnections++;
        cell1->connections[0].cell = cell2;
        cell1->connections[0].distance = desiredDistance;
        cell1->connections[0].angleFromPrevious = 360.0f;
        return true;
    }

    // *****
    // special case: cell1 has one connection
    // *****
    if (1 == cell1->numConnections) {
        auto connectedCellDelta = cell1->connections[0].cell->pos - cell1->pos;
        data.cellMap.correctDirection(connectedCellDelta);
        auto prevAngle = Math::angleOfVector(connectedCellDelta);
        auto angleDiff = newAngle - prevAngle;
        if (0 != desiredAngleOnCell1) {
            angleDiff = desiredAngleOnCell1;
        }
        angleDiff = Math::alignAngle(angleDiff, angleAlignment);
        if (angleDiff < 0) {
            angleDiff += 360.0;
        }
        angleDiff = Math::alignAngleOnBoundaries(angleDiff, 360.0f, angleAlignment);
        if (abs(angleDiff) < NEAR_ZERO || abs(angleDiff - 360.0f) < NEAR_ZERO || abs(angleDiff + 360.0f) < NEAR_ZERO) {
            return false;
        }

        cell1->connections[1].angleFromPrevious = angleDiff;
        cell1->connections[0].angleFromPrevious = 360.0f - angleDiff;

        cell1->numConnections++;
        cell1->connections[1].cell = cell2;
        cell1->connections[1].distance = desiredDistance;
        return true;
    }

    // *****
    // process general case
    // *****

    // find appropriate index for new connection
    int index = 0;
    float prevAngle = 0;
    float nextAngle = 0;
    for (; index < cell1->numConnections; ++index) {
        auto prevIndex = (index + cell1->numConnections - 1) % cell1->numConnections;
        prevAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell1->connections[prevIndex].cell->pos - cell1->pos));
        nextAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell1->connections[index].cell->pos - cell1->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, newAngle) || prevIndex == index) {
            break;
        }
    }

    // create new connection object
    CellConnection newConnection;
    newConnection.cell = cell2;
    newConnection.distance = desiredDistance;

    float angleFromPrevious;
    auto refAngle = cell1->connections[index].angleFromPrevious;
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

    // insert new connection to a clone of the existing connection array
    if (index == 0) {
        index = cell1->numConnections;  // connection at index 0 should be an invariant
    }
    for (int j = cell1->numConnections; j > index; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }
    cell1->connections[index] = newConnection;
    cell1->connections[(index + 1) % (cell1->numConnections + 1)].angleFromPrevious = refAngle - angleFromPrevious;
    if (cell1->signalRoutingRestriction.active && index == cell1->signalRoutingRestriction.refConnectionIndex) {
        ++cell1->signalRoutingRestriction.refConnectionIndex;
    }

    // align angles
    if (angleAlignment != ConstructorAngleAlignment_None) {
        auto const angleUnit = 360.0f / (angleAlignment + 1);
        for (int i = 0; i < cell1->numConnections + 1; ++i) {
            cell1->connections[i].angleFromPrevious = Math::alignAngle(cell1->connections[i].angleFromPrevious, angleAlignment);
            if (abs(cell1->connections[i].angleFromPrevious) < NEAR_ZERO) {
                cell1->connections[i].angleFromPrevious = angleUnit;
            }
        }

        for (int repetition = 0; repetition < MAX_CELL_BONDS; ++repetition) {
            float sumAngle = 0;
            for (int i = 0, j = cell1->numConnections + 1; i < j; ++i) {
                sumAngle += cell1->connections[i].angleFromPrevious;
            }
            if (sumAngle > 360.0f + NEAR_ZERO || sumAngle < 360.0f - NEAR_ZERO) {
                int indexWithMaxAngle = -1;
                float maxAngle = 0;
                for (int k = 0, l = cell1->numConnections + 1; k < l; ++k) {
                    if (cell1->connections[k].angleFromPrevious > maxAngle) {
                        maxAngle = cell1->connections[k].angleFromPrevious;
                        indexWithMaxAngle = k;
                    }
                }
                if (sumAngle > 360.0f + NEAR_ZERO) {
                    cell1->connections[indexWithMaxAngle].angleFromPrevious -= angleUnit;
                } else {
                    cell1->connections[indexWithMaxAngle].angleFromPrevious += angleUnit;
                }
            } else {
                break;
            }
        }
    }
    cell1->numConnections++;

    return true;
}

__inline__ __device__ void CellConnectionProcessor::deleteConnectionOneWay(Cell* cell1, Cell* cell2)
{
    for (int i = 0; i < cell1->numConnections; ++i) {
        if (cell1->connections[i].cell == cell2) {
            float angleToAdd = cell1->connections[i].angleFromPrevious;
            for (int j = i; j < cell1->numConnections - 1; ++j) {
                cell1->connections[j] = cell1->connections[j + 1];
            }

            if (i < cell1->numConnections - 1) {
                cell1->connections[i].angleFromPrevious += angleToAdd;
            } else {
                cell1->connections[0].angleFromPrevious += angleToAdd;
            }

            --cell1->numConnections;
            return;
        }
    }
}

__inline__ __device__ bool CellConnectionProcessor::existCrossingConnections(SimulationData& data, float2 pos1, float2 pos2, int detached, int color)
{
    auto distance = Math::length(pos1 - pos2);
    if (distance > cudaSimulationParameters.cellMaxBindingDistance[color]) {
        return false;
    }

    bool result = false;
    data.cellMap.executeForEach((pos1 + pos2) / 2, distance, detached, [&](auto const& otherCell) {
        if ((otherCell->pos.x == pos1.x && otherCell->pos.y == pos1.y) || (otherCell->pos.x == pos2.x && otherCell->pos.y == pos2.y)) {
            return;
        }
        if (otherCell->tryLock()) {
            for (int i = 0; i < otherCell->numConnections; ++i) {
                if (Math::crossing(pos1, pos2, otherCell->pos, otherCell->connections[i].cell->pos)) {
                    otherCell->releaseLock();
                    result = true;
                    return;
                }
            }
            otherCell->releaseLock();
        }
    });
    return result;
}

__inline__ __device__ bool CellConnectionProcessor::wouldResultInOverlappingConnection(Cell* cell1, float2 otherCellPos)
{
    auto const& n = cell1->numConnections;
    if (n < 2) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        auto connectedCell = cell1->connections[i].cell;
        auto nextConnectedCell = cell1->connections[(i + 1) % n].cell;
        bool bothConnected = false;
        for (int j = 0; j < connectedCell->numConnections; ++j) {
            if (connectedCell->connections[j].cell == nextConnectedCell) {
                bothConnected = true;
                break;
            }
        }
        if (!bothConnected) {
            continue;
        }
        if (Math::crossing(cell1->pos, otherCellPos, connectedCell->pos, nextConnectedCell->pos)) {
            return true;
        }
    }
    return false;
}

__inline__ __device__ bool CellConnectionProcessor::isConnectedConnected(Cell* cell, Cell* otherCell)
{
    if (cell == otherCell) {
        return true;
    }
    bool result = false;
    for (int i = 0; i < otherCell->numConnections; ++i) {
        auto const& connectedCell = otherCell->connections[i].cell;
        if (connectedCell == cell) {
            result = true;
            break;
        }

        for (int j = 0; j < connectedCell->numConnections; ++j) {
            auto const& connectedConnectedCell = connectedCell->connections[j].cell;
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

__inline__ __device__ CellConnectionProcessor::ReferenceAndActualAngle
CellConnectionProcessor::calcLargestGapReferenceAndActualAngle(SimulationData& data, Cell* cell, float angleDeviation)
{
    if (0 == cell->numConnections) {
        return ReferenceAndActualAngle{0, data.numberGen1.random() * 360};
    }
    auto displacement = cell->connections[0].cell->pos - cell->pos;
    data.cellMap.correctDirection(displacement);
    auto angle = Math::angleOfVector(displacement);
    int index = 0;
    float largestAngleGap = 0;
    float angleOfLargestAngleGap = 0;
    auto numConnections = cell->numConnections;
    for (int i = 1; i <= numConnections; ++i) {
        auto angleDiff = cell->connections[i % numConnections].angleFromPrevious;
        if (angleDiff > largestAngleGap) {
            largestAngleGap = angleDiff;
            index = i % numConnections;
            angleOfLargestAngleGap = angle;
        }
        angle += angleDiff;
    }

    auto angleFromPrev = cell->connections[index].angleFromPrevious;
    for (int i = 0; i < numConnections - 1; ++i) {
        if (angleDeviation > angleFromPrev / 2) {
            angleDeviation -= angleFromPrev / 2;
            index = (index + 1) % numConnections;
            angleOfLargestAngleGap += angleFromPrev;
            angleFromPrev = cell->connections[index].angleFromPrevious;
            angleDeviation = angleDeviation - angleFromPrev / 2;
        }
        if (angleDeviation < -angleFromPrev / 2) {
            angleDeviation += angleFromPrev / 2;
            index = (index + numConnections - 1) % numConnections;
            angleFromPrev = cell->connections[index].angleFromPrevious;
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

__inline__ __device__ void CellConnectionProcessor::scheduleOperationOnCell(SimulationData& data, Cell* cell, int operationIndex)
{
    auto origOperationIndex = atomicCAS(&cell->scheduledOperationIndex, -1, operationIndex);
    for (int depth = 0; depth < MaxOperationsPerCell; ++depth) {
        if (origOperationIndex == -1) {
            break;
        }
        auto& origOperation = data.structuralOperations.at(origOperationIndex);
        origOperationIndex = atomicCAS(&origOperation.nextOperationIndex, -1, operationIndex);
    }
}

