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
    __inline__ __device__ static void processSignalStorage(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
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
    } else if (mode == MemoryMode_SignalStorage) {
        processSignalStorage(data, statistics, cell);
    }
}

__inline__ __device__ void MemoryProcessor::processIntegrator(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;

    // Initialize memory if necessary
    if (memory.numSignalEntries != 1) {
        memory.numSignalEntries = 1;
        memory.signalEntries = data.objects.heap.getTypedSubArray<SignalEntry>(1);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            memory.signalEntries->channels[i] = cell->signal.channels[i];
        }
    } else {
        auto const& newSignalWeight = memory.modeData.signalIntegrator.newSignalWeight;
        auto const& channelBitMask = memory.channelBitMask;
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            memory.signalEntries->channels[i] = (1.0f - newSignalWeight) * memory.signalEntries->channels[i] + newSignalWeight * cell->signal.channels[i];
            if (channelBitMask & (1 << i)) {
                cell->signal.channels[i] = memory.signalEntries->channels[i];
            }
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
    if (memory.numSignalEntries != signalDelay.delay) {
        memory.numSignalEntries = signalDelay.delay;
        memory.signalEntries = data.objects.heap.getTypedSubArray<SignalEntry>(memory.numSignalEntries);
        signalDelay.numSignalEntriesInitialized = 0;
        signalDelay.ringBufferIndex = 0;
    }

    // Store output from the oldest memory entry (at ringBufferIndex)
    Signal output;
    if (signalDelay.numSignalEntriesInitialized == memory.numSignalEntries) {
        auto ringBufferIndex = signalDelay.ringBufferIndex;
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            output.channels[k] = memory.signalEntries[ringBufferIndex].channels[k];
        }
    }

    // Store current signal at ringBufferIndex (this position contains the oldest entry which we just output)
    for (int k = 0; k < MAX_CHANNELS; ++k) {
        memory.signalEntries[signalDelay.ringBufferIndex].channels[k] = cell->signal.channels[k];
    }

    // Write output
    if (signalDelay.numSignalEntriesInitialized == memory.numSignalEntries) {
        auto const& channelBitMask = memory.channelBitMask;
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            if (channelBitMask & (1 << k)) {
                cell->signal.channels[k] = output.channels[k];
            }
        }
    }

    // Advance the ring buffer index to point to the next oldest entry
    signalDelay.ringBufferIndex = (signalDelay.ringBufferIndex + 1) % memory.numSignalEntries;

    // Track initialization progress
    if (signalDelay.numSignalEntriesInitialized < memory.numSignalEntries) {
        ++signalDelay.numSignalEntriesInitialized;
    }
}

__device__ __inline__ void MemoryProcessor::processSignalRecorder(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;
    auto& signalRecorder = memory.modeData.signalRecorder;

    if (memory.numSignalEntries == 0) {
        return;
    }

    auto& state = signalRecorder.state;
    auto& numWrittenSignalEntries = signalRecorder.numWrittenSignalEntries;
    auto& numReadSignalEntries = signalRecorder.numReadSignalEntries;

    // Validate and correct numRecorded if out of bounds
    if (numWrittenSignalEntries > memory.numSignalEntries) {
        numWrittenSignalEntries = 0;
    }

    // Validate and correct numReadSignalEntries if out of bounds
    if (numReadSignalEntries >= numWrittenSignalEntries || numReadSignalEntries >= memory.numSignalEntries) {
        numReadSignalEntries = 0;
    }

    // State machine for recording/reading
    if (state == SignalRecorderState_Idle) {
        // Check channel[0] to initiate recording or reading
        if (cell->signal.channels[Channels::MemoryReadWriteAction] > TRIGGER_THRESHOLD && !signalRecorder.readOnly) {
            // Reset numRecorded to start fresh
            numWrittenSignalEntries = 0;
            state = SignalRecorderState_Recording;
        } else if (
            cell->signal.channels[Channels::MemoryReadWriteAction] < -TRIGGER_THRESHOLD
            || (signalRecorder.readOnly && abs(cell->signal.channels[Channels::MemoryReadWriteAction]) > TRIGGER_THRESHOLD)) {
            // Start reading - reset numReadSignalEntries
            numReadSignalEntries = 0;
            state = SignalRecorderState_Reading;
        }
    }

    if (state == SignalRecorderState_Recording) {
        // Record signal to memory at index numRecorded
        if (numWrittenSignalEntries < memory.numSignalEntries) {
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                memory.signalEntries[numWrittenSignalEntries].channels[k] = cell->signal.channels[k];
            }
            ++numWrittenSignalEntries;
        }
        // Recording complete when numRecorded reaches numSignalEntries
        if (numWrittenSignalEntries >= memory.numSignalEntries) {
            state = SignalRecorderState_Idle;
        }
    } else if (state == SignalRecorderState_Reading) {
        // Read recorded memory entry at index numReadSignalEntries
        if (numReadSignalEntries < numWrittenSignalEntries) {
            auto const& channelBitMask = memory.channelBitMask;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                if (channelBitMask & (1 << k)) {
                    cell->signal.channels[k] = memory.signalEntries[numReadSignalEntries].channels[k];
                }
            }
            ++numReadSignalEntries;
        }
        // Reading complete when numReadSignalEntries reaches numRecorded
        if (numReadSignalEntries >= numWrittenSignalEntries) {
            numReadSignalEntries = 0;
            state = SignalRecorderState_Idle;
        }
    }
}

__device__ __inline__ void MemoryProcessor::processSignalStorage(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& memory = cell->cellTypeData.memory;
    auto& signalStorage = memory.modeData.signalStorage;

    if (memory.numSignalEntries == 0) {
        return;
    }

    auto const& inputValue = cell->signal.channels[Channels::MemoryReadWriteAction];
    auto const numSignalEntries = toInt(memory.numSignalEntries);

    // Calculate the index based on |channel[0]| * (numSignalEntries - 1)
    auto absInput = abs(inputValue);
    auto index = toInt(absInput * toFloat(numSignalEntries - 1) + 0.5f);

    // Clamp index between 0 and numSignalEntries - 1
    if (index < 0) {
        index = 0;
    } else if (index >= numSignalEntries) {
        index = numSignalEntries - 1;
    }

    auto const& channelBitMask = memory.channelBitMask;
    if (signalStorage.readOnly) {
        // In readonly mode, always read regardless of sign
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            if (channelBitMask & (1 << k)) {
                cell->signal.channels[k] = memory.signalEntries[index].channels[k];
            }
        }
    } else if (inputValue >= 0) {
        // Read mode: channel[0] >= 0
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            if (channelBitMask & (1 << k)) {
                cell->signal.channels[k] = memory.signalEntries[index].channels[k];
            }
        }
    } else {
        // Write mode: channel[0] < 0
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            memory.signalEntries[index].channels[k] = cell->signal.channels[k];
        }
    }
}
