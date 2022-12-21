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
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Nerve];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;

        int inputExecutionOrderNumber;
        auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);

        auto const& nerve = cell->cellFunctionData.nerve;
        if (nerve.pulseMode > 0 && (data.timestep % (cudaSimulationParameters.cellMaxExecutionOrderNumbers * nerve.pulseMode) == cell->executionOrderNumber)) {
            if (nerve.alternationMode == 0) {
                activity.channels[0] += 1;
            } else {
                auto evenPulse = data.timestep % (cudaSimulationParameters.cellMaxExecutionOrderNumbers * nerve.pulseMode * nerve.alternationMode * 2)
                    < cell->executionOrderNumber + cudaSimulationParameters.cellMaxExecutionOrderNumbers * nerve.pulseMode * nerve.alternationMode;
                activity.channels[0] += evenPulse ? 1 : -1;
            }
        }
        CellFunctionProcessor::setActivity(cell, activity);
    }
}
