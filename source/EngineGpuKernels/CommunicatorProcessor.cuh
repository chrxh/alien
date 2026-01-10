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

    __inline__ __device__ static void evaluateLastMatches(SimulationData& data, Cell* cell);
    __inline__ __device__ static void
    addNearestMatch(SimulationData& data, Cell* cell, Cell* otherCell, int& numNearestMatches, float2* nearestMatches, float* nearestMatchDistances, int& nearestMatchesLock);
    __inline__ __device__ static void
    mergeMatches(SimulationData& data, Cell* cell, int numNearestMatches, float2* nearestMatches, float* nearestMatchDistances);
    __inline__ __device__ static bool
    tryTransmitSignal(Cell* senderCell, Cell* receiverCell, int newNumTimesSent);

    static float constexpr ScanStep = 8.0f;
    static float constexpr MatchLookupRadius = 5.0f;
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
    __shared__ float seedAngle;
    __shared__ float range;
    __shared__ int maxTimesSent;
    __shared__ int currentNumTimesSent;

    __shared__ int numNearestMatches; 
    __shared__ float2 nearestMatches[MAX_SENDER_MATCHES];
    __shared__ float nearestMatchDistances[MAX_SENDER_MATCHES];
    __shared__ int nearestMatchesLock;

    if (threadIdx.x == 0) {
        auto& sender = cell->cellTypeData.communicator.modeData.sender;
        range = sender.range;
        maxTimesSent = sender.maxTimesSent;
        currentNumTimesSent = cell->signal.numTimesSent;
        seedAngle = data.primaryNumberGen.random(360.0f);
        numNearestMatches = 0;
        nearestMatchesLock = 0;
    }
    __syncthreads();

    // Check if signal can still be forwarded
    if (currentNumTimesSent >= maxTimesSent) {
        return;
    }

    // Evaluate last matches in parallel: remove out-of-range ones and transmit to remaining ones
    evaluateLastMatches(data, cell);
    __syncthreads();

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
                addNearestMatch(data, cell, otherCell, numNearestMatches, nearestMatches, nearestMatchDistances, nearestMatchesLock);
            }
            otherCell = otherCell->nextCell;
        }
    }
    __syncthreads();
    
    // Update last matches in sender: merge existing lastMatches with newly found nearestMatches
    if (threadIdx.x == 0) {
        mergeMatches(data, cell, numNearestMatches, nearestMatches, nearestMatchDistances);
    }
}

__inline__ __device__ void
CommunicatorProcessor::addNearestMatch(SimulationData& data, Cell* cell, Cell* otherCell, int& numNearestMatches, float2* nearestMatches, float* nearestMatchDistances, int& nearestMatchesLock)
{
    auto matchDistance = data.cellMap.getDistance(cell->pos, otherCell->pos);
    
    // Acquire lock
    while (atomicExch_block(&nearestMatchesLock, 1) == 1) {}
    __threadfence_block();
    
    // Check if we can add this match
    if (numNearestMatches < MAX_SENDER_MATCHES) {
        // Array not full, just add it
        nearestMatches[numNearestMatches] = otherCell->pos;
        nearestMatchDistances[numNearestMatches] = matchDistance;
        ++numNearestMatches;
    } else {
        // Array is full, find farthest match and replace if this one is closer
        int farthestIdx = 0;
        float farthestDist = nearestMatchDistances[0];
        for (int i = 1; i < MAX_SENDER_MATCHES; ++i) {
            if (nearestMatchDistances[i] > farthestDist) {
                farthestDist = nearestMatchDistances[i];
                farthestIdx = i;
            }
        }
        if (matchDistance < farthestDist) {
            nearestMatches[farthestIdx] = otherCell->pos;
            nearestMatchDistances[farthestIdx] = matchDistance;
        }
    }
    
    // Release lock
    __threadfence_block();
    atomicExch_block(&nearestMatchesLock, 0);
}

__inline__ __device__ void
CommunicatorProcessor::mergeMatches(SimulationData& data, Cell* cell, int numNearestMatches, float2* nearestMatches, float* nearestMatchDistances)
{
    auto& sender = cell->cellTypeData.communicator.modeData.sender;
    
    // Create combined array of all matches with distances
    float2 allMatches[MAX_SENDER_MATCHES * 2];
    float allDistances[MAX_SENDER_MATCHES * 2];
    int numAllMatches = 0;
    
    // Add existing lastMatches (already filtered for range in evaluateLastMatches)
    for (int i = 0; i < sender.numLastMatches; ++i) {
        allMatches[numAllMatches] = sender.lastMatches[i];
        allDistances[numAllMatches] = data.cellMap.getDistance(cell->pos, sender.lastMatches[i]);
        ++numAllMatches;
    }
    
    // Add new nearestMatches (only if at least MatchLookupRadius distance to existing ones)
    for (int i = 0; i < numNearestMatches; ++i) {
        bool isTooClose = false;
        for (int j = 0; j < numAllMatches; ++j) {
            if (data.cellMap.getDistance(nearestMatches[i], allMatches[j]) < MatchLookupRadius) {
                isTooClose = true;
                break;
            }
        }
        if (!isTooClose) {
            allMatches[numAllMatches] = nearestMatches[i];
            allDistances[numAllMatches] = nearestMatchDistances[i];
            ++numAllMatches;
        }
    }
    
    // Sort by distance (simple bubble sort for small array)
    for (int i = 0; i < numAllMatches - 1; ++i) {
        for (int j = 0; j < numAllMatches - i - 1; ++j) {
            if (allDistances[j] > allDistances[j + 1]) {
                // Swap
                float tempDist = allDistances[j];
                allDistances[j] = allDistances[j + 1];
                allDistances[j + 1] = tempDist;
                float2 tempPos = allMatches[j];
                allMatches[j] = allMatches[j + 1];
                allMatches[j + 1] = tempPos;
            }
        }
    }
    
    // Store the nearest MAX_SENDER_MATCHES matches
    sender.numLastMatches = min(numAllMatches, MAX_SENDER_MATCHES);
    for (int i = 0; i < sender.numLastMatches; ++i) {
        sender.lastMatches[i] = allMatches[i];
    }
}

__inline__ __device__ void CommunicatorProcessor::evaluateLastMatches(SimulationData& data, Cell* cell)
{
    auto& sender = cell->cellTypeData.communicator.modeData.sender;
    float senderRange = sender.range;
    int newNumTimesSent = cell->signal.numTimesSent + 1;
    
    __shared__ int numValidMatches;
    __shared__ float2 validMatches[MAX_SENDER_MATCHES];
    
    if (threadIdx.x == 0) {
        numValidMatches = 0;
    }
    __syncthreads();
    
    // Process matches in parallel - each thread handles different matches
    for (int i = threadIdx.x, numLastMatches = sender.numLastMatches; i < numLastMatches; i += blockDim.x) {
        float2 matchPos = sender.lastMatches[i];
        float distance = data.cellMap.getDistance(cell->pos, matchPos);
        
        if (distance <= senderRange) {
            // Match is still in range - look for receiver within MatchLookupRadius
            Cell* foundCells[MAX_SENDER_MATCHES];
            int numFoundCells = 0;
            data.cellMap.getMatchingCells(foundCells, MAX_SENDER_MATCHES, numFoundCells, matchPos, MatchLookupRadius, cell->detached,
                [&](Cell* otherCell) {
                    return otherCell->cellType == CellType_Communicator &&
                           otherCell->cellTypeData.communicator.mode == CommunicatorMode_Receiver &&
                           !cell->isSameCreature(otherCell);
                });
            
            if (numFoundCells > 0) {
                // Found a receiver - transmit and keep this match with updated position
                Cell* otherCell = foundCells[0];
                tryTransmitSignal(cell, otherCell, newNumTimesSent);
                
                // Add to valid matches atomically
                int idx = atomicAdd(&numValidMatches, 1);
                if (idx < MAX_SENDER_MATCHES) {
                    validMatches[idx] = otherCell->pos;
                }
            }
            // If no cell found, the match is discarded
        }
    }
    __syncthreads();
    
    // Copy valid matches back to sender (done by thread 0)
    if (threadIdx.x == 0) {
        sender.numLastMatches = min(numValidMatches, MAX_SENDER_MATCHES);
        for (int i = 0; i < sender.numLastMatches; ++i) {
            sender.lastMatches[i] = validMatches[i];
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
