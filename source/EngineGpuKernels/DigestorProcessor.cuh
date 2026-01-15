#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Entity.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DigestorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DigestorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Digestor];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void DigestorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    auto convertedEnergy = object->rawEnergy;
    auto threshold = (1.0f - object->cellTypeData.digestor.rawEnergyConductivity) * cudaSimulationParameters.maxRawEnergyConversion.value[object->color];
    convertedEnergy = min(convertedEnergy, threshold);

    object->rawEnergy -= convertedEnergy;
    object->usableEnergy += convertedEnergy;
}
