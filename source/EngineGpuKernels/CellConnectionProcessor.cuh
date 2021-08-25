#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"

class CellConnectionProcessor
{
public:
    __inline__ __device__ static void scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDelConnections(SimulationData& data, Cell* cell);
    __inline__ __device__ static void scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex);

    __inline__ __device__ static void processConnectionsOperations(SimulationData& data);
    __inline__ __device__ static void processDelCellOperations(SimulationData& data);

    __inline__ __device__ static void addConnectionsForConstructor(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float desiredAngleOnCell1,
        float desiredAngleOnCell2,
        float desiredDistance,
        int angleAlignment = 0);
    __inline__ __device__ static void delConnectionsForConstructor(Cell* cell1, Cell* cell2);

private:
    __inline__ __device__ static void addConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void addConnection(
        SimulationData& data,
        Cell* cell1,
        Cell* cell2,
        float2 const& posDelta,
        float desiredDistance,
        float desiredAngleOnCell1 = 0,
        int angleAlignment = 0);

    __inline__ __device__ static void delConnections(Cell* cell);
    __inline__ __device__ static void delConnection(Cell* cell1, Cell* cell2);

    __inline__ __device__ static void delCell(SimulationData& data, Cell* cell, int cellIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void
CellConnectionProcessor::scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2)
{
    auto index = atomicAdd(data.numOperations, 1);
    if (index < data.entities.cellPointers.getNumEntries()) {
        Operation& operation = data.operations[index];
        operation.type = Operation::Type::AddConnections; 
        operation.data.addConnectionOperation.cell = cell1;
        operation.data.addConnectionOperation.otherCell = cell2;
    } else {
        atomicSub(data.numOperations, 1);
    }
}


__inline__ __device__ void CellConnectionProcessor::scheduleDelConnections(SimulationData& data, Cell* cell)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
        auto index = atomicAdd(data.numOperations, 1);
        if (index < data.entities.cellPointers.getNumEntries()) {
            Operation& operation = data.operations[index];
            operation.type = Operation::Type::DelConnections;
            operation.data.delConnectionsOperation.cell = cell;
        } else {
            atomicSub(data.numOperations, 1);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex)
{
    auto index = atomicAdd(data.numOperations, 1);
    if (index < data.entities.cellPointers.getNumEntries()) {
        Operation& operation = data.operations[index];
        operation.type = Operation::Type::DelCell;
        operation.data.delCellOperation.cell = cell;
        operation.data.delCellOperation.cellIndex = cellIndex;
    } else {
        atomicSub(data.numOperations, 1);
    }
}

__inline__ __device__ void CellConnectionProcessor::processConnectionsOperations(SimulationData& data)
{
    auto partition = calcPartition(*data.numOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.operations[index];
        if (Operation::Type::DelConnections == operation.type) {
            delConnections(operation.data.delConnectionsOperation.cell);
        }
        if (Operation::Type::AddConnections == operation.type) {
            addConnections(
                data, operation.data.addConnectionOperation.cell, operation.data.addConnectionOperation.otherCell);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::processDelCellOperations(SimulationData& data)
{
    auto partition = calcPartition(*data.numOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.operations[index];
        if (Operation::Type::DelCell == operation.type) {
            delCell(data, operation.data.delCellOperation.cell, operation.data.delCellOperation.cellIndex);
        }
    }
}

__inline__ __device__ void CellConnectionProcessor::addConnectionsForConstructor(
    SimulationData& data,
    Cell* cell1,
    Cell* cell2,
    float desiredAngleOnCell1,
    float desiredAngleOnCell2,
    float desiredDistance,
    int angleAlignment)
{
    auto posDelta = cell2->absPos - cell1->absPos;
    data.cellMap.mapDisplacementCorrection(posDelta);
    addConnection(data, cell1, cell2, posDelta, desiredDistance, desiredAngleOnCell1, angleAlignment);
    addConnection(data, cell2, cell1, posDelta * (-1), desiredDistance, desiredAngleOnCell2, angleAlignment);
}

__inline__ __device__ void
CellConnectionProcessor::delConnectionsForConstructor(Cell* cell1, Cell* cell2)
{
    delConnection(cell1, cell2);
    delConnection(cell2, cell1);
}

__inline__ __device__ void CellConnectionProcessor::addConnections(SimulationData& data, Cell* cell1, Cell* cell2)
{
    SystemDoubleLock lock;
    lock.init(&cell1->locked, &cell2->locked);
    if (lock.tryLock()) {
        __threadfence();

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
            data.cellMap.mapDisplacementCorrection(posDelta);
            addConnection(data, cell1, cell2, posDelta, Math::length(posDelta));
            addConnection(data, cell2, cell1, posDelta * (-1), Math::length(posDelta));
        }

        __threadfence();
        lock.releaseLock();
    }
}

__inline__ __device__ void CellConnectionProcessor::addConnection(
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
        data.cellMap.mapDisplacementCorrection(connectedCellDelta);
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

    auto lastConnectedCellDelta = cell1->connections[0].cell->absPos - cell1->absPos;
    data.cellMap.mapDisplacementCorrection(lastConnectedCellDelta);
    float angle = Math::angleOfVector(lastConnectedCellDelta);

    int i = 1;
    while (true) {
        auto nextAngle = angle + cell1->connections[i].angleFromPrevious;

        if ((angle < newAngle && newAngle <= nextAngle)
            || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
            break;
        }

        ++i;
        if (i == cell1->numConnections) {
            i = 0;
        }
        angle = nextAngle;
        if (angle > 360.0f) {
            angle -= 360.0f;
        }
    }

    CellConnection newConnection;
    newConnection.cell = cell2;
    newConnection.distance = desiredDistance;

    auto angleDiff1 = newAngle - angle;
    if (angleDiff1 < 0) {
        angleDiff1 += 360.0f;
    }
    auto angleDiff2 = cell1->connections[i].angleFromPrevious;
    if (angleDiff1 > angleDiff2) {
        angleDiff1 = angleDiff2;
    }

    auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
    if (0 == desiredAngleOnCell1) {
        newConnection.angleFromPrevious = angleDiff2 * factor;
    } else {
        if (desiredAngleOnCell1 > angleDiff2) {
            desiredAngleOnCell1 = angleDiff2;
        }
        newConnection.angleFromPrevious = desiredAngleOnCell1;
    }
    newConnection.angleFromPrevious = Math::alignAngle(newConnection.angleFromPrevious, angleAlignment % 7);

    for (int j = cell1->numConnections; j > i; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }
    cell1->connections[i] = newConnection;
    ++cell1->numConnections;

    cell1->connections[(++i) % cell1->numConnections].angleFromPrevious = angleDiff2 - newConnection.angleFromPrevious;
}

__inline__ __device__ void CellConnectionProcessor::delConnections(Cell* cell)
{
    if (cell->tryLock()) {
        for (int i = cell->numConnections - 1; i >= 0; --i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->tryLock()) {

                delConnection(cell, connectedCell);
                delConnection(connectedCell, cell);

                connectedCell->releaseLock();
            }
        }
        cell->releaseLock();
    }
}

__inline__ __device__ void CellConnectionProcessor::delConnection(Cell* cell1, Cell* cell2)
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
            cell->energy = 0;

            data.entities.cellPointers.at(cellIndex) = nullptr;
        }

        cell->releaseLock();
    }
}
