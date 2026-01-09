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

    static float constexpr ScanStep = 8.0f;
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
    __shared__ bool isActive;
    if (threadIdx.x == 0) {
        isActive = cell->signalState == SignalState_Active;
    }
    __syncthreads();

    if (!isActive) {
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
    // TODO:
    // call here a (to be created) method which evaluates the last matches in cell->cellTypeData.communicator.modeData.sender.lastMatches
    // It should:
    // - remove the elements from lastMatches from the arrays which are out of range
    //   (which means the the distance between matchPos and cell via data.cellMap.getDistance is above cell->cellTypeData.communicator.modeData.sender.range)
    // - invoke tryTransmitSignal on each remaining last matches
    
    __shared__ float seedAngle;
    __shared__ float range;
    __shared__ int maxTimesSent;
    __shared__ int currentNumTimesSent;

    __shared__ int numNearestMatches; 
    __shared__ float2 nearestMatches[MAX_SENDER_MATCHES];
    __shared__ int nearestMatchesLock = 0; 

    if (threadIdx.x == 0) {
        auto& sender = cell->cellTypeData.communicator.modeData.sender;
        range = sender.range;
        maxTimesSent = sender.maxTimesSent;
        currentNumTimesSent = cell->signal.numTimesSent;
        seedAngle = data.primaryNumberGen.random(360.0f);
        numNearestMatches = 0;
    }
    __syncthreads();

    // Check if signal can still be forwarded
    if (currentNumTimesSent >= maxTimesSent) {
        return;
    }

    auto const senderPos = cell->pos;

    auto isMatch = [&cell](Cell* otherCell){

        // Must be a communicator in receiver mode
        if (otherCell->cellType == CellType_Communicator && otherCell->cellTypeData.communicator.mode == CommunicatorMode_Receiver) {

            // Must be from a different creature
            if (!cell->isSameCreature(otherCell)) {
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
            }
        }
        return false;
    };

    // Use ray scanning
    auto angle = 360.0f * toFloat(threadIdx.x) / toFloat(blockDim.x) + seedAngle;

    for (float distance = ScanStep; distance <= range; distance += ScanStep) {
        auto delta = Math::unitVectorOfAngle(angle) * distance;
        auto scanPos = senderPos + delta;
        data.cellMap.correctPosition(scanPos);

        auto otherCell = data.cellMap.getFirst(scanPos);
        while (otherCell != nullptr) {
            if (isMatch(otherCell)) {
                tryTransmitSignal(cell, otherCell, currentNumTimesSent + 1);

                // Store nearest matches
                // 
                // TODO:
                // nearestMatches should contain the nearest matches (maximum MAX_SENDER_MATCHES)
                // near with respect to the distance between cell and otherCell (via data.cellMap.getDistance)
                // numNearestMatches encodes the length of nearestMatches (should always be <= MAX_SENDER_MATCHES) 
            }
            otherCell = otherCell->nextCell;
        }
    }
    __syncthreads();
    
    // Update last matches in sender
    // 
    // TODO:
    // calculate the nearest matches in cell->cellTypeData.communicator.modeData.sender.lastMatches and nearestMatches and store the results in
    // cell->cellTypeData.communicator.modeData.sender.lastMatches (maximum MAX_SENDER_MATCHES) 
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
