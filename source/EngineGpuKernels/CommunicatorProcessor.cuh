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
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void CommunicatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void CommunicatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (cell->signalState != SignalState_Active) {
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
    auto& sender = cell->cellTypeData.communicator.modeData.sender;

    // Check if signal can still be forwarded
    if (cell->signal.numTimesSent >= sender.maxTimesSent) {
        return;
    }

    auto const senderPos = cell->pos;
    auto const range = sender.range;
    auto const newNumTimesSent = cell->signal.numTimesSent + 1;

    // Reset lastMatches for this sender
    sender.numLastMatches = 0;

    // Search for nearby receivers from different creatures
    data.cellMap.executeForEach(senderPos, range, cell->detached, [&](Cell* otherCell) {
        // Must be a communicator in receiver mode
        if (otherCell->cellType != CellType_Communicator) {
            return;
        }
        if (otherCell->cellTypeData.communicator.mode != CommunicatorMode_Receiver) {
            return;
        }

        // Must be from a different creature
        if (cell->isSameCreature(otherCell)) {
            return;
        }

        auto const& receiver = otherCell->cellTypeData.communicator.modeData.receiver;

        // Check color restriction
        if (receiver.restrictToColor != 255 && cell->color != receiver.restrictToColor) {
            return;
        }

        // Check lineage restriction
        if (receiver.restrictToLineage != LineageRestriction_No) {
            if (cell->creature == nullptr || otherCell->creature == nullptr) {
                return;
            }
            if (receiver.restrictToLineage == LineageRestriction_SameLineage) {
                if (cell->creature->lineageId != otherCell->creature->lineageId) {
                    return;
                }
            } else if (receiver.restrictToLineage == LineageRestriction_OtherLineage) {
                if (cell->creature->lineageId == otherCell->creature->lineageId) {
                    return;
                }
            }
        }

        // Copy signal to receiver with incremented numTimesSent
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            otherCell->signal.channels[k] = cell->signal.channels[k];
        }
        otherCell->signal.numTimesSent = newNumTimesSent;
        otherCell->signalState = SignalState_Active;

        // Track this match in lastMatches
        if (sender.numLastMatches < MAX_SENDER_MATCHES) {
            sender.lastMatches[sender.numLastMatches] = otherCell->pos;
            sender.numLastMatches++;
        }
    });
}
