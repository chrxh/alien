#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"

class NerveProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void NerveProcessor::process(SimulationData& data)
{
    auto partition = calcAllThreadsPartition(data.cellFunctionOperations[Enums::CellFunction_Nerve].getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto operation = data.cellFunctionOperations[Enums::CellFunction_Nerve].at(i);
        auto cell = operation.cell;
        bool inputAvailable;
        CellFunctionProcessor::calcInputActivity(cell, inputAvailable, cell->activity);
    }
}
