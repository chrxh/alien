#pragma once

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "Map.cuh"
#include "ObjectFactory.cuh"
#include "Physics.cuh"
#include "SpotCalculator.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

class CellFunctionProcessor
{
public:
    __inline__ __device__ static void collectCellFunctionOperations(SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CellFunctionProcessor::collectCellFunctionOperations(SimulationData& data)
{
    auto& cells = data.objects.cellPointers;
    auto partition = calcAllThreadsPartition(cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto& cell = cells.at(index);


    }
}
