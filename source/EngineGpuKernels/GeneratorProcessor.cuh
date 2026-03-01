#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "Base.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "SimulationStatistics.cuh"
#include "TOs.cuh"

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
        auto const& object = operation.object;

        auto& generator = object->typeData.cell.cellTypeData.generator;

        // Calculate current position within the period, shifted by timeOffset
        int period = (generator.mode == GeneratorMode_SquareSignal) ? generator.modeData.squareSignal.period : generator.modeData.sawtoothSignal.period;
        if (period <= 0) {
            continue;
        }
        auto timestepInPeriod = (generator.numPulses + generator.timeOffset) % period;
        float outputValue = 0.0f;

        if (generator.mode == GeneratorMode_SquareSignal) {
            auto& squareSignal = generator.modeData.squareSignal;
            auto halfPeriod = period / 2;

            if (timestepInPeriod < halfPeriod) {
                outputValue = squareSignal.amplitude;
            } else {
                outputValue = -squareSignal.amplitude;
            }
        } else if (generator.mode == GeneratorMode_SawtoothSignal) {
            auto& sawtoothSignal = generator.modeData.sawtoothSignal;

            // Linear increase from 0 to amplitude over the period
            outputValue = sawtoothSignal.amplitude * toFloat(timestepInPeriod) / toFloat(period);
        }
        outputValue += generator.valueOffset;

        // Set the output signal

        if (generator.additive) {
            object->typeData.cell.signal.channels[Channels::GeneratorOutput] += outputValue;
        } else {
            object->typeData.cell.signal.channels[Channels::GeneratorOutput] = outputValue;
        }

        // Clamp final signal to valid range [-2.0, 2.0]
        object->typeData.cell.signal.channels[Channels::GeneratorOutput] =
            max(-2.0f, min(2.0f, object->typeData.cell.signal.channels[Channels::GeneratorOutput]));

        // Increment timestep counter and wrap around at period
        generator.numPulses += TIMESTEPS_PER_CELL_FUNCTION;
        if (generator.numPulses >= period) {
            generator.numPulses -= period;
        }

        statistics.incNumGeneratorPulses(object->color);
    }
}
