#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Object.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class CommunicatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void processSender(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static bool
    tryTransmitSignal(Cell* senderCell, Cell* receiverCell, int newNumTimesSent);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void CommunicatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void CommunicatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ bool shouldProcess;
    if (threadIdx.x == 0) {
        // Process if signal is Active or Fading (signal has been or is being processed this timestep)
        shouldProcess = SignalProcessor::isManuallyTriggered(data, cell);
    }
    __syncthreads();

    if (!shouldProcess) {
        return;
    }

    auto const& mode = cell->cellTypeData.communicator.mode;
    if (mode == CommunicatorMode_Sender) {
        processSender(data, statistics, cell);
    }
    // Receiver mode: signals are set by senders, no additional processing needed
}

__device__ __inline__ void CommunicatorProcessor::processSender(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    __shared__ float range;
    __shared__ int maxTimesSent;
    __shared__ int currentNumTimesSent;
    __shared__ float2 senderPos;

    if (threadIdx.x == 0) {
        auto& sender = cell->cellTypeData.communicator.modeData.sender;
        range = sender.range;
        maxTimesSent = sender.maxTimesSent;
        currentNumTimesSent = cell->signal.numTimesSent;
        senderPos = cell->pos;
    }
    __syncthreads();

    // Check if signal can still be forwarded
    if (currentNumTimesSent >= maxTimesSent) {
        return;
    }

    auto const newNumTimesSent = currentNumTimesSent + 1;
    int rangeInt = static_cast<int>(ceilf(range));

    // Matching lambda to check if a cell is a valid receiver
    auto isMatch = [&cell](Cell* otherCell) {
        // Must be a communicator in receiver mode
        if (otherCell->cellType != CellType_Communicator || otherCell->cellTypeData.communicator.mode != CommunicatorMode_Receiver) {
            return false;
        }

        // Must be from a different creature
        if (cell->isSameCreature(otherCell)) {
            return false;
        }

        auto const& receiver = otherCell->cellTypeData.communicator.modeData.receiver;

        // Check color restriction
        if (receiver.restrictToColor != 255 && cell->color != receiver.restrictToColor) {
            return false;
        }

        // Check lineage restriction
        if (receiver.restrictToLineage != LineageRestriction_No) {
            if (cell->creature == nullptr || otherCell->creature == nullptr) {
                return false;
            } else if (receiver.restrictToLineage == LineageRestriction_SameLineage) {
                if (cell->creature->lineageId != otherCell->creature->lineageId) {
                    return false;
                }
            } else if (receiver.restrictToLineage == LineageRestriction_OtherLineage) {
                if (cell->creature->lineageId == otherCell->creature->lineageId) {
                    return false;
                }
            }
        }

        return true;
    };

    // Calculate the total number of scan positions in the [-range, range] x [-range, range] region
    int diameter = 2 * rangeInt + 1;
    int totalPositions = diameter * diameter;

    // Each thread scans different positions in parallel
    for (int idx = threadIdx.x; idx < totalPositions; idx += blockDim.x) {
        int dx = (idx % diameter) - rangeInt;
        int dy = (idx / diameter) - rangeInt;

        // Check if within circular range
        float distSq = static_cast<float>(dx * dx + dy * dy);
        if (distSq > range * range) {
            continue;
        }

        float2 scanPos = {senderPos.x + static_cast<float>(dx), senderPos.y + static_cast<float>(dy)};
        data.cellMap.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto otherCell = data.cellMap.getFirst(scanPos);
        while (otherCell != nullptr) {
            if (isMatch(otherCell)) {
                tryTransmitSignal(cell, otherCell, newNumTimesSent);
            }
            otherCell = otherCell->nextCell;
        }
    }
}

__inline__ __device__ bool
CommunicatorProcessor::tryTransmitSignal(Cell* senderCell, Cell* receiverCell, int newNumTimesSent)
{
    receiverCell->getLock();

    // Check if we should override existing signal
    bool shouldTransmit = false;
    if (receiverCell->signalState != SignalState_Active) {
        shouldTransmit = true;
    } else if (newNumTimesSent < receiverCell->signal.numTimesSent) {
        // Override only if new signal has fewer transmission hops
        shouldTransmit = true;
    }

    if (shouldTransmit) {
        // Copy signal to receiver with incremented numTimesSent
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            receiverCell->signal.channels[k] = senderCell->signal.channels[k];
        }
        receiverCell->signal.numTimesSent = newNumTimesSent;
        receiverCell->signalState = SignalState_Active;
    }

    receiverCell->releaseLock();
    return shouldTransmit;
}
