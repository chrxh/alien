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

    __inline__ __device__ static Cell*
    findReceiver(SimulationData& data, Cell* senderCell, float2 const& scanPos, float absAngle, float distance);
    __inline__ __device__ static bool
    tryTransmitSignal(Cell* senderCell, Cell* receiverCell, int newNumTimesSent);

    __inline__ __device__ static uint64_t pack(float distance, float angle);
    __inline__ __device__ static void unpack(float& distance, float& angle, uint64_t bytes);

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
    __shared__ uint64_t lookupResults[MAX_SENDER_MATCHES];
    __shared__ float seedAngle;
    __shared__ float range;
    __shared__ int maxTimesSent;
    __shared__ int currentNumTimesSent;

    if (threadIdx.x == 0) {
        auto& sender = cell->cellTypeData.communicator.modeData.sender;
        range = sender.range;
        maxTimesSent = sender.maxTimesSent;
        currentNumTimesSent = cell->signal.numTimesSent;
        seedAngle = data.primaryNumberGen.random(360.0f);
        
        // Reset lastMatches
        sender.numLastMatches = 0;
        
        for (int i = 0; i < MAX_SENDER_MATCHES; ++i) {
            lookupResults[i] = 0xffffffffffffffff;
        }
    }
    __syncthreads();

    // Check if signal can still be forwarded
    if (currentNumTimesSent >= maxTimesSent) {
        return;
    }

    auto const newNumTimesSent = currentNumTimesSent + 1;
    auto const senderPos = cell->pos;

    // Use ray scanning similar to SensorProcessor
    auto angle = 360.0f * toFloat(threadIdx.x) / toFloat(blockDim.x) + seedAngle;

    for (float distance = ScanStep; distance <= range; distance += ScanStep) {
        auto delta = Math::unitVectorOfAngle(angle) * distance;
        auto scanPos = senderPos + delta;
        data.cellMap.correctPosition(scanPos);

        auto receiverCell = findReceiver(data, cell, scanPos, angle, distance);
        if (receiverCell != nullptr) {
            auto packed = pack(distance, angle);
            
            // Try to add to results (atomic operation to handle thread competition)
            for (int slot = 0; slot < MAX_SENDER_MATCHES; ++slot) {
                auto old = atomicCAS(reinterpret_cast<unsigned long long*>(&lookupResults[slot]), 
                                     0xffffffffffffffff, 
                                     static_cast<unsigned long long>(packed));
                if (old == 0xffffffffffffffff) {
                    // Successfully claimed this slot
                    break;
                }
            }
            break;  // Found a receiver on this ray, stop scanning further
        }
    }

    __syncthreads();

    // Process found receivers (only thread 0)
    if (threadIdx.x == 0) {
        auto& sender = cell->cellTypeData.communicator.modeData.sender;
        
        for (int slot = 0; slot < MAX_SENDER_MATCHES; ++slot) {
            if (lookupResults[slot] == 0xffffffffffffffff) {
                continue;
            }

            float distance, angle;
            unpack(distance, angle, lookupResults[slot]);

            auto delta = Math::unitVectorOfAngle(angle) * distance;
            auto scanPos = senderPos + delta;
            data.cellMap.correctPosition(scanPos);

            // Find the receiver again and try to transmit
            auto receiverCell = findReceiver(data, cell, scanPos, angle, distance);
            if (receiverCell != nullptr) {
                if (tryTransmitSignal(cell, receiverCell, newNumTimesSent)) {
                    // Track this match
                    if (sender.numLastMatches < MAX_SENDER_MATCHES) {
                        sender.lastMatches[sender.numLastMatches] = receiverCell->pos;
                        sender.numLastMatches++;
                    }
                }
            }
        }
    }
}

__inline__ __device__ Cell*
CommunicatorProcessor::findReceiver(SimulationData& data, Cell* senderCell, float2 const& scanPos, float absAngle, float distance)
{
    auto otherCell = data.cellMap.getFirst(scanPos);
    while (otherCell != nullptr) {
        // Must be a communicator in receiver mode
        if (otherCell->cellType == CellType_Communicator && 
            otherCell->cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
            
            // Must be from a different creature
            if (!senderCell->isSameCreature(otherCell)) {
                auto const& receiver = otherCell->cellTypeData.communicator.modeData.receiver;

                // Check color restriction
                bool matches = true;
                if (receiver.restrictToColor != 255 && senderCell->color != receiver.restrictToColor) {
                    matches = false;
                }

                // Check lineage restriction
                if (matches && receiver.restrictToLineage != LineageRestriction_No) {
                    if (senderCell->creature == nullptr || otherCell->creature == nullptr) {
                        matches = false;
                    } else if (receiver.restrictToLineage == LineageRestriction_SameLineage) {
                        if (senderCell->creature->lineageId != otherCell->creature->lineageId) {
                            matches = false;
                        }
                    } else if (receiver.restrictToLineage == LineageRestriction_OtherLineage) {
                        if (senderCell->creature->lineageId == otherCell->creature->lineageId) {
                            matches = false;
                        }
                    }
                }

                if (matches) {
                    return otherCell;
                }
            }
        }
        otherCell = otherCell->nextCell;
    }
    return nullptr;
}

__inline__ __device__ bool
CommunicatorProcessor::tryTransmitSignal(Cell* senderCell, Cell* receiverCell, int newNumTimesSent)
{
    // Try to lock the receiver
    if (!receiverCell->tryLock()) {
        // Already locked by another sender, try again
        for (int retry = 0; retry < 10; ++retry) {
            if (receiverCell->tryLock()) {
                break;
            }
            if (retry == 9) {
                return false;  // Give up after retries
            }
        }
    }

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

__inline__ __device__ uint64_t CommunicatorProcessor::pack(float distance, float angle)
{
    uint32_t distanceEncoded = static_cast<uint32_t>(distance);
    int32_t angleEncoded = static_cast<int32_t>(angle * 100.0f);
    return (static_cast<uint64_t>(distanceEncoded) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(angleEncoded));
}

__inline__ __device__ void CommunicatorProcessor::unpack(float& distance, float& angle, uint64_t bytes)
{
    distance = toFloat(bytes >> 32);
    angle = toFloat(static_cast<int32_t>(bytes & 0xFFFFFFFF)) / 100.0f;
}
