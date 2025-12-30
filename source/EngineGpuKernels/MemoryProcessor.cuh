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

    __inline__ __device__ static void processIntegrator(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void processDelay(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void processSignalRecorder(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void MemoryProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Memory];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void MemoryProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (cell->signalState != SignalState_Active) {
        return;
    }
    auto const& mode = cell->cellTypeData.memory.mode;
    if (mode == MemoryMode_SignalDelay) {
        processDelay(data, statistics, cell);
    } else if (mode == MemoryMode_SignalIntegrator) {
        processIntegrator(data, statistics, cell);
    } else if (mode == MemoryMode_SignalRecorder) {
        processSignalRecorder(data, statistics, cell);
    }
}

__inline__ __device__ void MemoryProcessor::processIntegrator(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;

    // Initialize memory if necessary
    if (memory.numMemoryEntries != 1) {
        memory.numMemoryEntries = 1;
        memory.memoryEntries = data.objects.heap.getTypedSubArray<MemoryEntry>(1);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            memory.memoryEntries->channels[i] = cell->signal.channels[i];
        }
    } else {
        auto const& newSignalWeight = memory.modeData.signalIntegrator.newSignalWeight;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            memory.memoryEntries->channels[i] = (1.0f - newSignalWeight) * memory.memoryEntries->channels[i] + newSignalWeight * cell->signal.channels[i];
            cell->signal.channels[i] = memory.memoryEntries->channels[i];
        }
    }
}

__device__ __inline__ void MemoryProcessor::processDelay(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;
    auto& signalDelay = memory.modeData.signalDelay;

    if (signalDelay.delay == 0) {
        return;
    }

    // Initialize memory if necessary
    if (memory.numMemoryEntries != signalDelay.delay) {
        memory.numMemoryEntries = signalDelay.delay;
        memory.memoryEntries = data.objects.heap.getTypedSubArray<MemoryEntry>(memory.numMemoryEntries);
        signalDelay.numMemoryEntriesInitialized = 0;
        signalDelay.ringBufferIndex = 0;
    }

    // Store output from the oldest memory entry (at ringBufferIndex)
    Signal output;
    if (signalDelay.numMemoryEntriesInitialized == memory.numMemoryEntries) {
        auto ringBufferIndex = signalDelay.ringBufferIndex;
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            output.channels[k] = memory.memoryEntries[ringBufferIndex].channels[k];
        }
    }

    // Store current signal at ringBufferIndex (this position contains the oldest entry which we just output)
    for (int k = 0; k < MAX_CHANNELS; ++k) {
        memory.memoryEntries[signalDelay.ringBufferIndex].channels[k] = cell->signal.channels[k];
    }

    // Write output
    if (signalDelay.numMemoryEntriesInitialized == memory.numMemoryEntries) {
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            cell->signal.channels[k] = output.channels[k];
        }
    }

    // Advance the ring buffer index to point to the next oldest entry
    signalDelay.ringBufferIndex = (signalDelay.ringBufferIndex + 1) % memory.numMemoryEntries;

    // Track initialization progress
    if (signalDelay.numMemoryEntriesInitialized < memory.numMemoryEntries) {
        ++signalDelay.numMemoryEntriesInitialized;
    }
}

__device__ __inline__ void MemoryProcessor::processSignalRecorder(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;
    auto& signalRecorder = memory.modeData.signalRecorder;

    if (memory.numMemoryEntries == 0) {
        return;
    }

    auto& state = signalRecorder.state;
    auto& numRecorded = signalRecorder.numRecordedMemoryEntries;
    auto& currentReadIndex = signalRecorder.currentReadIndex;

    // Validate and correct numRecorded if out of bounds
    if (numRecorded > memory.numMemoryEntries) {
        numRecorded = 0;
    }

    // Validate and correct currentReadIndex if out of bounds
    if (currentReadIndex >= numRecorded || currentReadIndex >= memory.numMemoryEntries) {
        currentReadIndex = 0;
    }

    // State machine for recording/reading
    if (state == SignalRecorderState_Idle) {
        // Check channel[0] to initiate recording or reading
        if (cell->signal.channels[0] > 0) {
            // Start recording only if not readOnly
            if (!signalRecorder.readOnly) {
                // Reset numRecorded to start fresh
                numRecorded = 0;
                state = SignalRecorderState_Recording;
            }
        } else if (cell->signal.channels[0] < 0) {
            // Start reading - reset currentReadIndex
            currentReadIndex = 0;
            state = SignalRecorderState_Reading;
        }
    }

    if (state == SignalRecorderState_Recording) {
        // Record signal to memory at index numRecorded
        if (numRecorded < memory.numMemoryEntries) {
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                memory.memoryEntries[numRecorded].channels[k] = cell->signal.channels[k];
            }
            ++numRecorded;
        }
        // Recording complete when numRecorded reaches numMemoryEntries
        if (numRecorded >= memory.numMemoryEntries) {
            state = SignalRecorderState_Idle;
        }
    } else if (state == SignalRecorderState_Reading) {
        // Read recorded memory entry at index currentReadIndex
        if (currentReadIndex < numRecorded) {
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                cell->signal.channels[k] = memory.memoryEntries[currentReadIndex].channels[k];
            }
            ++currentReadIndex;
        }
        // Reading complete when currentReadIndex reaches numRecorded
        if (currentReadIndex >= numRecorded) {
            currentReadIndex = 0;
            state = SignalRecorderState_Idle;
        }
    }
}
