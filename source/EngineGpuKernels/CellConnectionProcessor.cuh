#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"
#include "SpotCalculator.cuh"

class CellConnectionProcessor
{
public:
    __inline__ __device__ static void scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2, bool addTokens);
    __inline__ __device__ static void scheduleDelConnections(SimulationData& data, Cell* cell);
    __inline__ __device__ static void scheduleDelConnection(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex);
    __inline__ __device__ static void scheduleDelCellAndConnections(SimulationData& data, Cell* cell, int cellIndex);

    __inline__ __device__ static void processConnectionsOperations(SimulationData& data);
    __inline__ __device__ static void processDelCellOperations(SimulationData& data);

    __inline__ __device__ static void addConnections(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float desiredAngleOnCell1,
        float desiredAngleOnCell2,
        float desiredDistance,
        int angleAlignment = 0);
    __inline__ __device__ static void delConnections(Cell* cell1, Cell* cell2);
    __inline__ __device__ static void delConnectionOneWay(Cell* cell1, Cell* cell2);

private:
    __inline__ __device__ static void addConnectionsIntern(SimulationData& data, Cell* cell1, Cell* cell2, bool addTokens);
    __inline__ __device__ static void addConnectionIntern(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredAngleOnCell1 = 0,
        int angleAlignment = 0);

    __inline__ __device__ static void delConnectionsIntern(Cell* cell);
    __inline__ __device__ static void delConnectionIntern(Cell* cell1, Cell* cell2);

    __inline__ __device__ static void delCell(SimulationData& data, Cell* cell, int cellIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void
CellConnectionProcessor::scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2, bool addTokens)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::AddConnections;
    operation.data.addConnectionOperation.cell = cell1;
    operation.data.addConnectionOperation.otherCell = cell2;
    operation.data.addConnectionOperation.addTokens = addTokens;
    data.structuralOperations.tryAddEntry(operation);
}


__inline__ __device__ void CellConnectionProcessor::scheduleDelConnections(SimulationData& data, Cell* cell)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelConnections;
    operation.data.delConnectionsOperation.cell = cell;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void
CellConnectionProcessor::scheduleDelConnection(SimulationData& data, Cell* cell1, Cell* cell2)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelConnection;
    operation.data.delConnectionOperation.cell1 = cell1;
    operation.data.delConnectionOperation.cell2 = cell2;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void CellConnectionProcessor::scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelCell;
    operation.data.delCellOperation.cell = cell;
    operation.data.delCellOperation.cellIndex = cellIndex;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void
CellConnectionProcessor::scheduleDelCellAndConnections(SimulationData& data, Cell* cell, int cellIndex)
{
    StructuralOperation operation;
    operation.type = StructuralOperation::Type::DelCellAndConnections;
    operation.data.delCellAndConnectionOperation.cell = cell;
    operation.data.delCellAndConnectionOperation.cellIndex = cellIndex;
    data.structuralOperations.tryAddEntry(operation);
}

__inline__ __device__ void CellConnectionProcessor::processConnectionsOperations(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.structuralOperations.getNumOrigEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::DelConnection == operation.type) {
            delConnectionIntern(operation.data.delConnectionOperation.cell1, operation.data.delConnectionOperation.cell2);
        }
        if (StructuralOperation::Type::DelConnections == operation.type) {
            delConnectionsIntern(operation.data.delConnectionsOperation.cell);
        }
        if (StructuralOperation::Type::DelCellAndConnections == operation.type) {
            delConnectionsIntern(operation.data.delConnectionsOperation.cell);

            scheduleDelCell(
                data,
                operation.data.delCellAndConnectionOperation.cell,
                operation.data.delCellAndConnectionOperation.cellIndex);
        }
        if (StructuralOperation::Type::AddConnections == operation.type) {
            addConnectionsIntern(
                data,
                operation.data.addConnectionOperation.cell,
                operation.data.addConnectionOperation.otherCell,
                operation.data.addConnectionOperation.addTokens);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::processDelCellOperations(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.structuralOperations.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.structuralOperations.at(index);
        if (StructuralOperation::Type::DelCell == operation.type) {
            delCell(data, operation.data.delCellOperation.cell, operation.data.delCellOperation.cellIndex);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::addConnections(
    SimulationData& data,
    Cell* cell1,
    Cell* cell2,
    float desiredAngleOnCell1,
    float desiredAngleOnCell2,
    float desiredDistance,
    int angleAlignment)
{
    auto posDelta = cell2->absPos - cell1->absPos;
    data.cellMap.correctDirection(posDelta);
    addConnectionIntern(data, cell1, cell2, posDelta, desiredDistance, desiredAngleOnCell1, angleAlignment);
    addConnectionIntern(data, cell2, cell1, posDelta * (-1), desiredDistance, desiredAngleOnCell2, angleAlignment);
}

__inline__ __device__ void
CellConnectionProcessor::delConnections(Cell* cell1, Cell* cell2)
{
    delConnectionOneWay(cell1, cell2);
    delConnectionOneWay(cell2, cell1);
}

__inline__ __device__ void
CellConnectionProcessor::addConnectionsIntern(SimulationData& data, Cell* cell1, Cell* cell2, bool addTokens)
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
            auto posDelta = cell2->absPos - cell1->absPos;
            data.cellMap.correctDirection(posDelta);
            addConnectionIntern(data, cell1, cell2, posDelta, Math::length(posDelta));
            addConnectionIntern(data, cell2, cell1, posDelta * (-1), Math::length(posDelta));

/*
            //align connections
            addConnectionIntern(data, cell1, cell2, posDelta, 1, 0, 6);
            addConnectionIntern(data, cell2, cell1, posDelta * (-1), 1, 0, 6);
*/

            if (addTokens) {
                EntityFactory factory;
                factory.init(&data);

                auto cellMinEnergy =
                    SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellMinEnergy, data, cell1->absPos);
                auto newTokenEnergy = cudaSimulationParameters.tokenMinEnergy * 1.5f;
                if (cell1->energy > cellMinEnergy + newTokenEnergy) {
                    auto token = factory.createToken(cell1, cell2);
                    token->energy = newTokenEnergy;
                    cell1->energy -= newTokenEnergy;
                }
                if (cell2->energy > cellMinEnergy + newTokenEnergy) {
                    auto token = factory.createToken(cell2, cell1);
                    token->energy = newTokenEnergy;
                    cell2->energy -= newTokenEnergy;
                }
            }
        }

        lock.releaseLock();
    }
}

__inline__ __device__ void CellConnectionProcessor::addConnectionIntern(
    SimulationData& data,
    Cell* cell1,
    Cell* cell2,
    float2 const& posDelta,
    float desiredDistance,
    float desiredAngleOnCell1,
    int angleAlignment)
{
    auto newAngle = Math::angleOfVector(posDelta);

    if (0 == cell1->numConnections) {
        cell1->numConnections++;
        cell1->connections[0].cell = cell2;
        cell1->connections[0].distance = desiredDistance;
        cell1->connections[0].angleFromPrevious = 360.0f;
        return;
    }
    if (1 == cell1->numConnections) {
        cell1->numConnections++;
        cell1->connections[1].cell = cell2;
        cell1->connections[1].distance = desiredDistance;

        auto connectedCellDelta = cell1->connections[0].cell->absPos - cell1->absPos;
        data.cellMap.correctDirection(connectedCellDelta);
        auto prevAngle = Math::angleOfVector(connectedCellDelta);
        auto angleDiff = newAngle - prevAngle;
        if (0 != desiredAngleOnCell1) {
            angleDiff = desiredAngleOnCell1;
        }
        angleDiff = Math::alignAngle(angleDiff, angleAlignment % 7);
        if (angleDiff >= 0) {
            cell1->connections[1].angleFromPrevious = angleDiff;
            cell1->connections[0].angleFromPrevious = 360.0f - angleDiff;
        } else {
            cell1->connections[1].angleFromPrevious = 360.0f + angleDiff;
            cell1->connections[0].angleFromPrevious = -angleDiff;
        }
        return;
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
        newConnection.angleFromPrevious = Math::alignAngle(newConnection.angleFromPrevious, angleAlignment % 7);
    }

    //add connection
    for (int j = cell1->numConnections; j > index; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }
    cell1->connections[index] = newConnection;
    ++cell1->numConnections;


    //adjust reference angle of next connection
    auto nextAngleFromPrevious = refAngle - newConnection.angleFromPrevious;
    auto nextAngleFromPreviousAligned = Math::alignAngle(nextAngleFromPrevious, angleAlignment % 7);
    auto angleDiff = nextAngleFromPreviousAligned - nextAngleFromPrevious;

    auto nextIndex = (index + 1) % cell1->numConnections;
    auto nextNextIndex = (index + 2) % cell1->numConnections;
    auto nextNextAngleFromPrevious = cell1->connections[nextNextIndex].angleFromPrevious;
    if (nextNextAngleFromPrevious - angleDiff >= 0.0f && nextNextAngleFromPrevious - angleDiff <= 360.0f) {
        cell1->connections[nextIndex].angleFromPrevious = nextAngleFromPreviousAligned;
        cell1->connections[nextNextIndex].angleFromPrevious = nextNextAngleFromPrevious - angleDiff;
    } else {
        cell1->connections[nextIndex].angleFromPrevious = nextAngleFromPrevious;
    }

}

__inline__ __device__ void CellConnectionProcessor::delConnectionsIntern(Cell* cell)
{
    for (int i = cell->numConnections - 1; i >= 0; --i) {
        auto connectedCell = cell->connections[0].cell;
        SystemDoubleLock lock;
        lock.init(&cell->locked, &connectedCell->locked);
        if (lock.tryLock()) {

            if (cell->numConnections > 0) {
                delConnectionOneWay(cell, connectedCell);
                delConnectionOneWay(connectedCell, cell);
            }

            lock.releaseLock();
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::delConnectionIntern(Cell* cell1, Cell* cell2)
{
    SystemDoubleLock lock;
    lock.init(&cell1->locked, &cell2->locked);
    if (lock.tryLock()) {
        delConnectionOneWay(cell1, cell2);
        delConnectionOneWay(cell2, cell1);
        lock.releaseLock();
    }
}

__inline__ __device__ void CellConnectionProcessor::delConnectionOneWay(Cell* cell1, Cell* cell2)
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

__inline__ __device__ void CellConnectionProcessor::delCell(SimulationData& data, Cell* cell, int cellIndex)
{
    if (cell->tryLock()) {

        if (0 == cell->numConnections && cell->energy != 0 /* && _data->entities.cellPointers.at(cellIndex) == cell*/) {
            EntityFactory factory;
            factory.init(&data);
            factory.createParticle(cell->energy, cell->absPos, cell->vel, {cell->metadata.color});
            cell->setDeleted();

            data.entities.cellPointers.at(cellIndex) = nullptr;
        }

        cell->releaseLock();
    }
}
