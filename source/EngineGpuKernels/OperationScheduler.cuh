#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"

class OperationScheduler
{
public:
    __inline__ __device__ static void scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDelConnections(SimulationData& data, Cell* cell);
    __inline__ __device__ static void scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex);

    __inline__ __device__ static void processDelOperations(SimulationData& data);
    __inline__ __device__ static void processOtherOperations(SimulationData& data);

private:
    __inline__ __device__ static void addConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void addConnection(SimulationData& data, Cell* cell1, Cell* cell2, float2 const& posDelta);

    __inline__ __device__ static void delConnections(Cell* cell);
    __inline__ __device__ static void delConnection(Cell* cell1, Cell* cell2);

    __inline__ __device__ static void delCell(SimulationData& data, Cell* cell, int cellIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void OperationScheduler::scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2)
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


__inline__ __device__ void OperationScheduler::scheduleDelConnections(SimulationData& data, Cell* cell)
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

__inline__ __device__ void OperationScheduler::scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex)
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

__inline__ __device__ void OperationScheduler::processDelOperations(SimulationData& data)
{
    auto partition = calcPartition(*data.numOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.operations[index];
        if (Operation::Type::DelConnections == operation.type) {
            delConnections(operation.data.delConnectionsOperation.cell);
        }
        if (Operation::Type::DelCell == operation.type) {
            delCell(data, operation.data.delCellOperation.cell, operation.data.delCellOperation.cellIndex);
        }
    }
}

__inline__ __device__ void OperationScheduler::processOtherOperations(SimulationData& data)
{
    auto partition = calcPartition(*data.numOperations, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& operation = data.operations[index];
        if (Operation::Type::AddConnections == operation.type) {
            addConnections(data, operation.data.addConnectionOperation.cell, operation.data.addConnectionOperation.otherCell);
        }
    }
}

__inline__ __device__ void OperationScheduler::addConnections(SimulationData& data, Cell* cell1, Cell* cell2)
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

        if (!alreadyConnected) {
            auto posDelta = cell2->absPos - cell1->absPos;
            data.cellMap.mapDisplacementCorrection(posDelta);
            addConnection(data, cell1, cell2, posDelta);
            addConnection(data, cell2, cell1, posDelta * (-1));
        }

        __threadfence();
        lock.releaseLock();
    }
}

__inline__ __device__ void
OperationScheduler::addConnection(SimulationData& data, Cell* cell1, Cell* cell2, float2 const& posDelta)
{
    auto newAngle = Math::angleOfVector(posDelta);

    int i = 0;
    float angle = 0;
    float prevAngle = 0;
    for (; i < cell1->numConnections; ++i) {
        auto connectedCell = cell1->connections[i].cell;
        auto connectedCellDelta = connectedCell->absPos - cell1->absPos;
        data.cellMap.mapDisplacementCorrection(connectedCellDelta);
        angle = Math::angleOfVector(connectedCellDelta);
        if (newAngle < angle) {
            break;
        }
        prevAngle = angle;
    }
    for (int j = cell1->numConnections; j > i; --j) {
        cell1->connections[j] = cell1->connections[j - 1];
    }

    cell1->numConnections++;
    cell1->connections[i].cell = cell2;
    cell1->connections[i].distance = Math::length(posDelta);

    if (0 == i && 1 < cell1->numConnections) {
        cell1->connections[1].angleFromPrevious = angle - newAngle;
    }
    if (0 < i && i + 1 == cell1->numConnections) {
        cell1->connections[i].angleFromPrevious = newAngle - angle;
    }
    if (0 < i && i + 1 < cell1->numConnections) {
        auto factor = (newAngle - prevAngle) / (angle - prevAngle);
        cell1->connections[i].angleFromPrevious = cell1->connections[i].angleFromPrevious * factor;
        cell1->connections[i + 1].angleFromPrevious = cell1->connections[i].angleFromPrevious * (1 - factor);
    }
}

__inline__ __device__ void OperationScheduler::delConnections(Cell* cell)
{
    if (cell->tryLock()) {
        for (int i = cell->numConnections - 1; i >= 0; --i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->tryLock()) {
                __threadfence();

                delConnection(cell, connectedCell);
                delConnection(connectedCell, cell);

                __threadfence();
                connectedCell->releaseLock();
            }
        }
        cell->releaseLock();
    }
}

__inline__ __device__ void OperationScheduler::delConnection(Cell* cell1, Cell* cell2)
{
    for (int i = 0; i < cell1->numConnections; ++i) {
        if (cell1->connections[i].cell == cell2) {
            if (i < cell1->numConnections - 1) {
                float angleToAdd = i > 0 ? cell1->connections[i].angleFromPrevious : 0;
                for (int j = i; j < cell1->numConnections - 1; ++j) {
                    cell1->connections[j] = cell1->connections[j + 1];
                }
                cell1->connections[i].angleFromPrevious += angleToAdd;
            }
            --cell1->numConnections;
            return;
        }
    }
}

__inline__ __device__ void OperationScheduler::delCell(SimulationData& data, Cell* cell, int cellIndex)
{
    if (cell->tryLock()) {
        __threadfence();

        if (0 == cell->numConnections && cell->energy != 0 /* && _data->entities.cellPointers.at(cellIndex) == cell*/) {
            EntityFactory factory;
            factory.init(&data);
            factory.createParticle(cell->energy, cell->absPos, cell->vel, {cell->metadata.color});
            cell->energy = 0;

            data.entities.cellPointers.at(cellIndex) = nullptr;
        }

        __threadfence();
        cell->releaseLock();
    }
}
