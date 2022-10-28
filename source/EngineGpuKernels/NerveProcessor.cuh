#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"

class NerveProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void NerveProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Neuron];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto operation = operations.at(i);
        auto cell = operation.cell;
        auto inputActivity = CellFunctionProcessor::calcInputActivity(cell);
        CellFunctionProcessor::setActivity(cell, inputActivity);
    }
}
