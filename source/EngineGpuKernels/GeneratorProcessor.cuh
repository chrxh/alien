#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationStatistics.cuh"
#include "TO.cuh"

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
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto const& operation = operations.at(i);
        auto const& object = operation.cell;

        auto& generator = object->cellTypeData.generator;
        if (SignalProcessor::isAutoTriggered(data, cell, max(1, generator.autoTriggerInterval))) {
            if (object->signalState != SignalState_Active) {
                SignalProcessor::createEmptySignal(cell);
            }
            statistics.incNumGeneratorPulses(object->color);
            if (generator.pulseType == GeneratorPulseType_Positive) {
                object->signal.channels[Channels::CellTypeActivation] += 1.0f;
            } else {
                object->signal.channels[Channels::CellTypeActivation] += generator.numPulses < generator.alternationInterval ? 1.0f : -1.0f;
            }
            ++generator.numPulses;
            if (generator.alternationInterval > 0 && generator.numPulses == generator.alternationInterval * 2) {
                generator.numPulses = 0;
            }
        }
    }
}
