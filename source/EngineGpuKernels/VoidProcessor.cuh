#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class VoidProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, int objectIndex);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void VoidProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& objects = data.entities.objects;
    auto partition = calcSystemThreadPartition(objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto& object = objects.at(index);
        if (!object) {
            continue;
        }
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (object->typeData.cell.cellType != CellType_Void) {
            continue;
        }
        if (object->typeData.cell.cellState != CellState_Ready) {
            continue;
        }
        if (object->typeData.cell.activationTime != 0) {
            continue;
        }
        processCell(data, result, object, index);
    }
}

__device__ __inline__ void VoidProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, int objectIndex)
{
    auto totalEnergy = object->typeData.cell.usableEnergy + object->typeData.cell.rawEnergy + object->typeData.cell.reservedEnergy;

    if (object->numConnections > 0) {
        auto energyPerNeighbor = totalEnergy / object->numConnections;
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if (connectedObject->type == ObjectType_Cell) {
                atomicAdd(&connectedObject->typeData.cell.usableEnergy, energyPerNeighbor);
            }
        }
    }

    object->typeData.cell.usableEnergy = 0;
    object->typeData.cell.rawEnergy = 0;
    object->typeData.cell.reservedEnergy = 0;

    ObjectConnectionProcessor::scheduleDeleteObject(data, objectIndex);
}
