#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class VoidProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void VoidProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Void];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void VoidProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto totalEnergy = object->typeData.cell.usableEnergy + object->typeData.cell.rawEnergy + object->typeData.cell.reservedEnergy;

    int cellNeighborCount = 0;
    for (int i = 0; i < object->numConnections; ++i) {
        if (object->connections[i].object->type == ObjectType_Cell) {
            ++cellNeighborCount;
        }
    }

    if (cellNeighborCount > 0) {
        auto energyPerNeighbor = totalEnergy / cellNeighborCount;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if (connectedObject->type == ObjectType_Cell) {
                atomicAdd(&connectedObject->typeData.cell.usableEnergy, energyPerNeighbor);
            }
        }
        object->typeData.cell.usableEnergy = 0;
        object->typeData.cell.rawEnergy = 0;
        object->typeData.cell.reservedEnergy = 0;
    }

    object->typeData.cell.cellState = CellState_InstantDying;
}
