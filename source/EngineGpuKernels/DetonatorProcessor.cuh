#pragma once


#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "EngineInterface/CellFunctionConstants.h"
#include "Object.cuh"
#include "ParticleProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DetonatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DetonatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Reconnector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (activity.channels[0] >= abs(cudaSimulationParameters.cellFunctionDetonatorActivityThreshold)) {
    }
    CellFunctionProcessor::setActivity(cell, activity);
}
