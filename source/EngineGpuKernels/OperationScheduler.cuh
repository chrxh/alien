#pragma once

#include "Base.cuh"
#include "Definitions.cuh"
#include "SimulationData.cuh"
#include "ConstantMemory.cuh"

class OperationScheduler
{
public:
    __inline__ __device__ static void scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2);
    __inline__ __device__ static void scheduleDelConnections(SimulationData& data, Cell* cell, int cellIndex);
    __inline__ __device__ static void scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void OperationScheduler::scheduleAddConnections(SimulationData& data, Cell* cell1, Cell* cell2)
{
    auto index = atomicAdd(data.numAddConnectionOperations, 1);
    if (index < data.entities.cellPointers.getNumEntries()) {
        auto& operation = data.addConnectionOperations[index];
        operation.cell = cell1;
        operation.otherCell = cell2;
    } else {
        atomicSub(data.numAddConnectionOperations, 1);
    }
}


__inline__ __device__ void OperationScheduler::scheduleDelConnections(SimulationData& data, Cell* cell, int cellIndex)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
        auto index = atomicAdd(data.numDelOperations, 1);
        if (index < data.entities.cellPointers.getNumEntries()) {
            auto& operation = data.delOperations[index];
            operation.type = DelOperation::Type::DelConnections;
            operation.cell = cell;
            operation.cellIndex = cellIndex;
        } else {
            atomicSub(data.numDelOperations, 1);
        }
    }
}

__inline__ __device__ void OperationScheduler::scheduleDelCell(SimulationData& data, Cell* cell, int cellIndex)
{
    auto index = atomicAdd(data.numDelOperations, 1);
    if (index < data.entities.cellPointers.getNumEntries()) {
        auto& operation = data.delOperations[index];
        operation.type = DelOperation::Type::DelCell;
        operation.cell = cell;
        operation.cellIndex = cellIndex;
    } else {
        atomicSub(data.numDelOperations, 1);
    }
}
