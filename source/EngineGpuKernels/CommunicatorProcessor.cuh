#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "Entities.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"
#include "ObjectConnectionProcessor.cuh"

class CommunicatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void processSender(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void tryTransmitSignal(SimulationData& data, Object* senderObject, Object* receiverObject, int newNumTimesSent);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void CommunicatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void CommunicatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ bool shouldProcess;
    if (threadIdx.x == 0) {
        shouldProcess = NeuronProcessor::isManuallyTriggered(data, object);
    }
    __syncthreads();

    if (!shouldProcess) {
        return;
    }

    auto const& mode = object->typeData.cell.cellTypeData.communicator.mode;
    if (mode == CommunicatorMode_Sender) {
        processSender(data, statistics, object);
    }
    // Receiver mode: signals are set by senders, no additional processing needed
}

__device__ __inline__ void CommunicatorProcessor::processSender(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ float range;
    __shared__ int maxTimesSent;
    __shared__ int currentNumTimesSent;
    __shared__ float2 senderPos;

    if (threadIdx.x == 0) {
        auto& sender = object->typeData.cell.cellTypeData.communicator.modeData.sender;
        range = sender.range;
        maxTimesSent = sender.maxTimesSent;
        currentNumTimesSent = object->typeData.cell.signal.numTimesSent;
        senderPos = object->pos;
    }
    __syncthreads();

    // Check if signal can still be forwarded
    if (currentNumTimesSent >= maxTimesSent) {
        return;
    }

    auto const newNumTimesSent = currentNumTimesSent + 1;
    int rangeInt = static_cast<int>(ceilf(range));

    // Matching lambda to check if a cell is a valid receiver
    auto isMatch = [&object](Object* otherObject) {
        // Must be a communicator in receiver mode
        if (otherObject->type != ObjectType_Cell) {
            return false;
        }
        if (otherObject->typeData.cell.cellType != CellType_Communicator || otherObject->typeData.cell.cellTypeData.communicator.mode != CommunicatorMode_Receiver) {
            return false;
        }

        // Must be from a different creature
        if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
            return false;
        }

        auto const& receiver = otherObject->typeData.cell.cellTypeData.communicator.modeData.receiver;

        // Check color restriction
        if (receiver.restrictToColor != 255 && object->color != receiver.restrictToColor) {
            return false;
        }

        // Check lineage restriction
        if (receiver.restrictToLineage != LineageRestriction_No) {
            if (receiver.restrictToLineage == LineageRestriction_RelatedLineage) {
                if (!object->typeData.cell.creature->genome->isRelatedLineage(otherObject->typeData.cell.creature->genome)) {
                    return false;
                }
            } else if (receiver.restrictToLineage == LineageRestriction_UnrelatedLineage) {
                if (object->typeData.cell.creature->genome->isRelatedLineage(otherObject->typeData.cell.creature->genome)) {
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
        float distSquared = static_cast<float>(dx * dx + dy * dy);
        if (distSquared > range * range) {
            continue;
        }

        float2 scanPos = {senderPos.x + static_cast<float>(dx), senderPos.y + static_cast<float>(dy)};
        data.objectMap.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto otherObject = data.objectMap.getFirst(scanPos);
        while (otherObject != nullptr) {
            if (isMatch(otherObject)) {
                tryTransmitSignal(data, object, otherObject, newNumTimesSent);
            }
            otherObject = otherObject->nextObject;
        }
    }
}

__inline__ __device__ void CommunicatorProcessor::tryTransmitSignal(SimulationData& data, Object* senderObject, Object* receiverObject, int newNumTimesSent)
{
    receiverObject->getLock();

    // Copy signal to receiver with incremented numTimesSent
    copyChannels(receiverObject->typeData.cell.signal.channels, senderObject->typeData.cell.signal.channels);
    receiverObject->typeData.cell.signal.numTimesSent = newNumTimesSent;

    // Translate angle in channel[1] from sender's reference direction to receiver's reference direction
    // The angle is encoded as value/180 degrees, where 1.0 = 180 deg and -1.0 = -180 deg
    // We need to maintain the absolute direction: senderRefDir rotated by senderAngle = receiverRefDir rotated by receiverAngle
    // Therefore: receiverAngle = senderAngle + (senderRefAngle - receiverRefAngle)
    auto senderRefDir = ObjectConnectionProcessor::calcReferenceDirection(data, senderObject);
    auto receiverRefDir = ObjectConnectionProcessor::calcReferenceDirection(data, receiverObject);
    auto senderRefAngle = Math::angleOfVector(senderRefDir);
    auto receiverRefAngle = Math::angleOfVector(receiverRefDir);
    auto angleDiff = senderRefAngle - receiverRefAngle;

    // The signal angle is encoded as angle/180, so the diff must also be scaled
    auto senderAngle = senderObject->typeData.cell.signal.channels[Channels::CommunicatorAngle];
    auto translatedAngle = senderAngle + angleDiff / 180.0f;
    // Normalize to [-1, 1] range (representing [-180, 180] degrees)
    translatedAngle = Math::getNormalizedAngle(translatedAngle * 180.0f, -180.0f) / 180.0f;
    receiverObject->typeData.cell.signal.channels[Channels::CommunicatorAngle] = translatedAngle;

    receiverObject->releaseLock();
}
