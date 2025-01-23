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
    auto& operations = data.cellTypeOperations[CellType_Oscillator];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;

        auto& oscillator = cell->cellTypeData.oscillator;
        if (SignalProcessor::isAutoTriggered(data, cell, max(1, oscillator.autoTriggerInterval))) {
            if (!cell->signal.active) {
                SignalProcessor::createEmptySignal(cell);
            }
            statistics.incNumOscillatorPulses(cell->color);
            if (oscillator.alternationInterval == 0) {
                cell->signal.channels[0] += 1.0f;
            } else {
                cell->signal.channels[0] += oscillator.numPulses < oscillator.alternationInterval ? 1.0f : -1.0f;
            }
            ++oscillator.numPulses;
            if (oscillator.alternationInterval > 0 && oscillator.numPulses == oscillator.alternationInterval * 2) {
                oscillator.numPulses = 0;  
            }
        }

    }
}
