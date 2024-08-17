#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "SimulationStatistics.cuh"

class NerveProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void NerveProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Nerve];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;

        auto activity = CellFunctionProcessor::calcInputActivity(cell);
        CellFunctionProcessor::updateInvocationState(cell, activity);

        auto const& nerve = cell->cellFunctionData.nerve;
        auto counter = (cell->age / cudaSimulationParameters.cellNumExecutionOrderNumbers) * cudaSimulationParameters.cellNumExecutionOrderNumbers
            + cell->executionOrderNumber % cudaSimulationParameters.cellNumExecutionOrderNumbers;
        if (nerve.pulseMode > 0 && (counter % (cudaSimulationParameters.cellNumExecutionOrderNumbers * nerve.pulseMode) == cell->executionOrderNumber)) {
            statistics.incNumNervePulses(cell->color);
            if (nerve.alternationMode == 0) {
                activity.channels[0] += 1.0f;
            } else {
                auto evenPulse = counter
                        % (cudaSimulationParameters.cellNumExecutionOrderNumbers * nerve.pulseMode * nerve.alternationMode * 2)
                    < cell->executionOrderNumber + cudaSimulationParameters.cellNumExecutionOrderNumbers * nerve.pulseMode * nerve.alternationMode;
                activity.channels[0] += evenPulse ? 1.0f : -1.0f;
            }
        }
        CellFunctionProcessor::setActivity(cell, activity);
    }
}
