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

    __inline__ __device__ static bool existCrossingConnections(SimulationData& data, float2 const& pos, float const& radius, bool detached);
    __inline__ __device__ static bool checkConnectedCellsForCrossingConnection(Cell* cell1, float2 otherCellPos);
    __inline__ __device__ static bool hasAngleSpace(SimulationData& data, Cell* cell, float angle, ConstructorAngleAlignment angleAlignment);
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

        if (!alreadyConnected && cell1->numConnections < MAX_CELL_BONDS && cell2->numConnections < MAX_CELL_BONDS) {

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

    if (checkConnectedCellsForCrossingConnection(cell1, cell2->pos)) {
        return false;
    }

    //Cell* nearCells[MAX_CELL_BONDS * 4];
    //int numNearCells;
    //data.cellMap.getMatchingCells(nearCells, MAX_CELL_BONDS * 4, numNearCells, cell1->pos, desiredDistance, cell1->detached, [&](Cell* const& nearCell) {
    //    return nearCell != cell1 && nearCell != cell2;
    //});

    //// evaluate candidates (locking is needed for the evaluation)
    //bool crossingLinks = false;
    //for (int j = 0; j < numNearCells; ++j) {
    //    auto nearCell = nearCells[j];
    //    if (nearCell->tryLock()) {
    //        for (int k = 0; k < nearCell->numConnections; ++k) {
    //            if (Math::crossing(cell1->pos, cell2->pos, nearCell->pos, nearCell->connections[k].cell->pos)) {
    //                crossingLinks = true;
    //            }
    //        }
    //        nearCell->releas eLock();
    //    }
    //}
    //if (crossingLinks) {
    //    //return false;
    //}

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

    // insert new connection
    if (index == 0) {
        index = cell1->numConnections;  // connection at index 0 should be an invariant
    }
    if (index == 0) {
        index = cell1->numConnections;  // connection at index 0 should be an invariant
    }
    for (int j = cell1->numConnections; j > index; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }
    cell1->connections[index] = newConnection;
    cell1->connections[(index + 1) % (cell1->numConnections + 1)].angleFromPrevious = refAngle - angleFromPrevious;

    // alignment
    if (angleAlignment != ConstructorAngleAlignment_None) {
        auto const angleUnit = 360.0f / toFloat(angleAlignment + 1);

        // align angles
        auto angleAdded = 0.0f;
        for (int i = 0; i < cell1->numConnections + 2; ++i) {
            auto index = i % (cell1->numConnections + 1);
            cell1->connections[index].angleFromPrevious = Math::alignAngle(cell1->connections[index].angleFromPrevious, angleAlignment);
            if (angleAdded > NEAR_ZERO && cell1->connections[index].angleFromPrevious - angleAdded > NEAR_ZERO) {
                cell1->connections[index].angleFromPrevious -= angleUnit;
            }
            if (abs(cell1->connections[index].angleFromPrevious) < NEAR_ZERO) {
                cell1->connections[index].angleFromPrevious = angleUnit;
                angleAdded = angleUnit;
            } else {
                angleAdded = 0;
            }
        }

        // ensure that sum of angles is 360 deg
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

__inline__ __device__ bool CellConnectionProcessor::existCrossingConnections(SimulationData& data, float2 const& pos, float const& radius, bool detached)
{
    auto result = false;
    data.cellMap.executeForEach(pos, radius, detached, [&](auto const& nearCell) {
        if (!nearCell->tryLock()) {
            return;
        }
        for (int j = 0; j < nearCell->numConnections; ++j) {
            auto const& connection = nearCell->connections[j];
            if (Math::crossing(pos, nearCell->pos, connection.cell->pos, connection.cell->pos + Math::unitVectorOfAngle(connection.angleFromPrevious))) {
                nearCell->releaseLock();
                result = true;
                return;
            }
        }
        nearCell->releaseLock();
    });
    return result;


    //Cell* nearCells[MAX_CELL_BONDS * 4];
    //int numNearCells;
    //data.cellMap.getMatchingCells(nearCells, MAX_CELL_BONDS * 4, numNearCells, pos, radius, false, [&](Cell* const&) { return true; });

    //for (int i = 0; i < numNearCells; ++i) {
    //    auto nearCell = nearCells[i];
    //    if (!nearCell->tryLock()) {
    //        continue;
    //    }
    //    for (int j = 0; j < nearCell->numConnections; ++j) {
    //        auto const& connection = nearCell->connections[j];
    //        if (Math::crossing(pos, nearCell->pos, connection.cell->pos, connection.cell->pos + Math::unitVectorOfAngle(connection.angleFromPrevious))) {
    //            nearCell->releaseLock();
    //            return true;
    //        }
    //    }
    //    nearCell->releaseLock();
    //}
    //return false;
}

__inline__ __device__ bool CellConnectionProcessor::checkConnectedCellsForCrossingConnection(Cell* cell1, float2 otherCellPos)
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

__inline__ __device__ bool CellConnectionProcessor::hasAngleSpace(SimulationData& data, Cell* cell, float angle, ConstructorAngleAlignment angleAlignment)
{
    if (angleAlignment == ConstructorAngleAlignment_None) {
        return true;
    }

    int index = 0;
    float prevAngle;
    float nextAngle;
    for (; index < cell->numConnections; ++index) {
        auto prevIndex = (index + cell->numConnections - 1) % cell->numConnections;
        prevAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell->connections[prevIndex].cell->pos - cell->pos));
        nextAngle = Math::angleOfVector(data.cellMap.getCorrectedDirection(cell->connections[index].cell->pos - cell->pos));
        if (Math::isAngleInBetween(prevAngle, nextAngle, angle) || prevIndex == index) {
            auto const angleUnit = 360.0f / toFloat(angleAlignment + 1);
            return cell->connections[index].angleFromPrevious > angleUnit + NEAR_ZERO;
        }
    }
    return true;
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

