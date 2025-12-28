#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Object.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class MemoryProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void MemoryProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Memory];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void MemoryProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;

    if (memory.mode == MemoryMode_SignalIntegrator) {
        // SignalIntegrator mode: integrate incoming signal with stored memory using weighted average
        // newStoredValue = newSignalWeight * incomingSignal + (1 - newSignalWeight) * oldStoredValue
        if (cell->signalState == SignalState_Active && memory.numMemoryEntries > 0) {
            float newSignalWeight = memory.modeData.signalIntegrator.newSignalWeight;
            float oldSignalWeight = 1.0f - newSignalWeight;

            auto& entry = memory.memoryEntries[0];
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                entry.channels[i] = newSignalWeight * cell->signal.channels[i] + oldSignalWeight * entry.channels[i];
            }
        }
    }
}
