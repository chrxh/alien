#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Object.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DigestorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DigestorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Digestor];
    auto partition = calcSystemThreadPartitionNew(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void DigestorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto convertedEnergy = cell->rawEnergy;
    auto threshold = (1.0f - cell->cellTypeData.digestor.rawEnergyConductivity) * cudaSimulationParameters.maxRawEnergyConversion.value[cell->color];
    convertedEnergy = min(convertedEnergy, threshold);

    cell->rawEnergy -= convertedEnergy;
    cell->usableEnergy += convertedEnergy;
}
