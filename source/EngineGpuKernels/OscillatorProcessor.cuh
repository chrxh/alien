#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationStatistics.cuh"

class OscillatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void OscillatorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Oscillator];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;

        auto const& oscillator = cell->cellFunctionData.oscillator;
        if (oscillator.pulseMode > 0 && cell->age % oscillator.pulseMode == 0) {
            if (!cell->signal.active) {
                SignalProcessor::createEmptySignal(cell);
            }
            statistics.incNumOscillatorPulses(cell->color);
            if (oscillator.alternationMode == 0) {
                cell->signal.channels[0] += 1.0f;
            } else {
                auto evenPulse = cell->age % (oscillator.pulseMode * oscillator.alternationMode * 2) < oscillator.pulseMode * oscillator.alternationMode;
                cell->signal.channels[0] += evenPulse ? 1.0f : -1.0f;
            }
        }

    }
}
