#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "ObjectTO.cuh"
#include "Base.cuh"
#include "CellConnectionProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationStatistics.cuh"

class GeneratorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void GeneratorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Generator];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;

        auto& generator = cell->cellTypeData.generator;
        if (SignalProcessor::isAutoTriggered(data, cell, max(1, generator.autoTriggerInterval))) {
            if (!cell->signal.active) {
                SignalProcessor::createEmptySignal(cell);
            }
            statistics.incNumGeneratorPulses(cell->color);
            if (generator.pulseType == GeneratorPulseType_Positive) {
                cell->signal.channels[Channels::GeneratorPulse] += 1.0f;
            } else {
                cell->signal.channels[Channels::GeneratorPulse] += generator.numPulses < generator.alternationInterval ? 1.0f : -1.0f;
            }
            ++generator.numPulses;
            if (generator.alternationInterval > 0 && generator.numPulses == generator.alternationInterval * 2) {
                generator.numPulses = 0;  
            }
        }

    }
}
