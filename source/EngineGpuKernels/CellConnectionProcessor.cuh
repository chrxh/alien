#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"
#include "ObjectFactory.cuh"

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

    __inline__ __device__
        static bool existCrossingConnections(SimulationData& data, float2 pos1, float2 pos2, int detached);

private:
    static int constexpr MaxOperationsPerCell = 30;

    __inline__ __device__ static bool scheduleOperationOnCell(SimulationData& data, Cell* cell, int operationIndex);

    __inline__ __device__ static void lockAndtryAddConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static bool tryAddConnectionOneWay(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredAngleOnCell1 = 0,
        ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_None);
    __inline__ __device__ static bool wouldResultInOverlappingConnection(Cell* cell1, Cell* cell2);
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
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectedCell = cell->connections[i].cell;
        {
            StructuralOperation operation;
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedCell = cell;
            operation.nextOperationIndex = -1;
            auto operationIndex = data.structuralOperations.tryAddEntry(operation);
            if (operationIndex != -1) {
                scheduleOperationOnCell(data, connectedCell, operationIndex);
            } else {
                CUDA_THROW_NOT_IMPLEMENTED();
            }
        }
        {
            StructuralOperation operation;
            operation.type = StructuralOperation::Type::DelConnection;
            operation.data.delConnection.connectedCell = connectedCell;
            operation.nextOperationIndex = -1;
            auto operationIndex = data.structuralOperations.tryAddEntry(operation);
            if (operationIndex != -1) {
                scheduleOperationOnCell(data, cell, operationIndex);
            } else {
                CUDA_THROW_NOT_IMPLEMENTED();
            }
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
    auto operationIndex = data.structuralOperations.tryAddEntry(operation);
    if (operationIndex == -1) {
        CUDA_THROW_NOT_IMPLEMENTED();
    }
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
                ObjectFactory factory;
                factory.init(&data);
                factory.createParticle(origCell->energy, origCell->absPos, origCell->vel, origCell->color);

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
    auto partition = calcAllThreadsPartition(data.objects.cellPointers.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = data.objects.cellPointers.at(index);
        if (!cell) {
            continue;
        }
        auto scheduledOperationIndex = cell->scheduledOperationIndex;
        if (scheduledOperationIndex != -1) {
            for (int depth = 0; depth < MaxOperationsPerCell; ++depth) {
                auto operation = data.structuralOperations.at(scheduledOperationIndex);
                switch (operation.type) {
                case StructuralOperation::Type::DelConnection: {
                    deleteConnectionOneWay(cell, operation.data.delConnection.connectedCell);
                } break;
                case StructuralOperation::Type::DelAllConnections: {
                    cell->numConnections = 0;
                } break;
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
    auto posDelta = cell2->absPos - cell1->absPos;
    data.cellMap.correctDirection(posDelta);

    CellConnection origConnections[MAX_CELL_BONDS];
    int origNumConnection = cell1->numConnections;
    for (int i = 0; i < origNumConnection; ++i) {
        origConnections[i] = cell1->connections[i];
    }

    if (!tryAddConnectionOneWay(data, cell1, cell2, posDelta, desiredDistance, desiredAngleOnCell1, angleAlignment)) {
        return false;
    }
    if(!tryAddConnectionOneWay(data, cell2, cell1, posDelta * (-1), desiredDistance, desiredAngleOnCell2, angleAlignment)) {
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
    if (wouldResultInOverlappingConnection(cell1, cell2)) {
        return false;
    }

    auto newAngle = Math::angleOfVector(posDelta);
    if (desiredDistance == 0) {
        desiredDistance = Math::length(posDelta);
    }

    if (0 == cell1->numConnections) {
        cell1->numConnections++;
        cell1->connections[0].cell = cell2;
        cell1->connections[0].distance = desiredDistance;
        cell1->connections[0].angleFromPrevious = 360.0f;
        return true;
    }
    if (1 == cell1->numConnections) {
        auto connectedCellDelta = cell1->connections[0].cell->absPos - cell1->absPos;
        data.cellMap.correctDirection(connectedCellDelta);
        auto prevAngle = Math::angleOfVector(connectedCellDelta);
        auto angleDiff = newAngle - prevAngle;
        if (0 != desiredAngleOnCell1) {
            angleDiff = desiredAngleOnCell1;
        }
        angleDiff = Math::alignAngle(angleDiff, angleAlignment % ConstructorAngleAlignment_Count);
        if (abs(angleDiff) < NEAR_ZERO || abs(angleDiff - 360.0f) < NEAR_ZERO || abs(angleDiff + 360.0f) < NEAR_ZERO) {
            return false;
        }

        if (angleDiff >= 0) {
            cell1->connections[1].angleFromPrevious = angleDiff;
            cell1->connections[0].angleFromPrevious = 360.0f - angleDiff;
        } else {
            cell1->connections[1].angleFromPrevious = 360.0f + angleDiff;
            cell1->connections[0].angleFromPrevious = -angleDiff;
        }

        cell1->numConnections++;
        cell1->connections[1].cell = cell2;
        cell1->connections[1].distance = desiredDistance;
        return true;
    }

    //find appropriate index for new connection
    int index = 0;
    float prevAngle = 0;
    float nextAngle = 0;
    for (; index < cell1->numConnections; ++index) {
        auto prevIndex = (index + cell1->numConnections - 1) % cell1->numConnections;
        prevAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell1->connections[prevIndex].cell->absPos - cell1->absPos));
        nextAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell1->connections[index].cell->absPos - cell1->absPos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, newAngle)) {
            break;
        }
    }

    //create connection object
    CellConnection newConnection;
    newConnection.cell = cell2;
    newConnection.distance = desiredDistance;
    newConnection.angleFromPrevious = 0;
    auto refAngle = cell1->connections[index].angleFromPrevious;
    if (Math::isAngleInBetween(prevAngle, nextAngle, newAngle)) {
        auto angleDiff1 = Math::subtractAngle(newAngle, prevAngle);
        auto angleDiff2 = Math::subtractAngle(nextAngle, prevAngle);
        auto factor = angleDiff2 != 0 ? angleDiff1 / angleDiff2 : 0.5f;
        if (0 == desiredAngleOnCell1) {
            newConnection.angleFromPrevious = refAngle * factor;
        } else {
            newConnection.angleFromPrevious = desiredAngleOnCell1;
        }
        newConnection.angleFromPrevious = min(newConnection.angleFromPrevious, refAngle);
        newConnection.angleFromPrevious = Math::alignAngle(newConnection.angleFromPrevious, angleAlignment % ConstructorAngleAlignment_Count);
    }
    if (newConnection.angleFromPrevious < NEAR_ZERO) {
        return false;
    }

    //adjust reference angle of next connection
    auto nextAngleFromPrevious = refAngle - newConnection.angleFromPrevious;
    auto nextAngleFromPreviousAligned = Math::alignAngle(nextAngleFromPrevious, angleAlignment % ConstructorAngleAlignment_Count);
    auto angleDiff = nextAngleFromPreviousAligned - nextAngleFromPrevious;

    auto nextIndex = index % cell1->numConnections;
    auto nextNextIndex = (index + 1) % cell1->numConnections;
    auto nextNextAngleFromPrevious = cell1->connections[nextNextIndex].angleFromPrevious;
    if (nextNextAngleFromPrevious - angleDiff >= 0.0f && nextNextAngleFromPrevious - angleDiff <= 360.0f) {
        if (nextAngleFromPreviousAligned < NEAR_ZERO || nextNextAngleFromPrevious - angleDiff < NEAR_ZERO) {
            return false;
        }
        cell1->connections[nextIndex].angleFromPrevious = nextAngleFromPreviousAligned;
        cell1->connections[nextNextIndex].angleFromPrevious = nextNextAngleFromPrevious - angleDiff;
    } else {
        if (nextAngleFromPrevious < NEAR_ZERO) {
            return false;
        }
        cell1->connections[nextIndex].angleFromPrevious = nextAngleFromPrevious;
    }

    //add connection
    for (int j = cell1->numConnections; j > index; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }
    cell1->connections[index] = newConnection;
    ++cell1->numConnections;

    return true;
}

__inline__ __device__ bool CellConnectionProcessor::wouldResultInOverlappingConnection(Cell* cell1, Cell* cell2)
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
        if (Math::crossing(cell1->absPos, cell2->absPos, connectedCell->absPos, nextConnectedCell->absPos)) {
            return true;
        }
    }
    return false;
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

__inline__ __device__ bool CellConnectionProcessor::existCrossingConnections(SimulationData& data, float2 pos1, float2 pos2, int detached)
{
    auto distance = Math::length(pos1 - pos2);
    if (distance > cudaSimulationParameters.cellMaxBindingDistance) {
        return false;
    }

    bool result = false;
    data.cellMap.executeForEach((pos1 + pos2) / 2, distance, detached, [&](auto const& otherCell) {
        if ((otherCell->absPos.x == pos1.x && otherCell->absPos.y == pos1.y) || (otherCell->absPos.x == pos2.x && otherCell->absPos.y == pos2.y)) {
            return;
        }
        if (otherCell->tryLock()) {
            for (int i = 0; i < otherCell->numConnections; ++i) {
                if (Math::crossing(pos1, pos2, otherCell->absPos, otherCell->connections[i].cell->absPos)) {
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

__inline__ __device__ bool CellConnectionProcessor::scheduleOperationOnCell(SimulationData& data, Cell* cell, int operationIndex)
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

