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

        auto signal = CellFunctionProcessor::updateFutureSignalOriginsAndReturnInputSignal(cell);

        auto const& nerve = cell->cellFunctionData.nerve;
        auto counter = (cell->age / cudaSimulationParameters.cellNumExecutionOrderNumbers) * cudaSimulationParameters.cellNumExecutionOrderNumbers
            + cell->executionOrderNumber % cudaSimulationParameters.cellNumExecutionOrderNumbers;
        if (nerve.pulseMode > 0 && cell->age % nerve.pulseMode == 0) {
            statistics.incNumNervePulses(cell->color);
            if (nerve.alternationMode == 0) {
                signal.channels[0] += 1.0f;
            } else {
                auto evenPulse = counter % (nerve.pulseMode * nerve.alternationMode * 2) < nerve.pulseMode * nerve.alternationMode;
                signal.channels[0] += evenPulse ? 1.0f : -1.0f;
            }
        }

        CellFunctionProcessor::setSignal(cell, signal);
    }
}
