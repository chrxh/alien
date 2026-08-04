#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "SimulationStatistics.cuh"

class CommunicatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void processSender(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static bool tryTransmitSignal(SimulationData& data, Object* senderObject, Object* receiverObject, float2 const& senderFacing);
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
    __shared__ int range;
    __shared__ float2 senderPos;
    __shared__ bool senderFrontValid;
    __shared__ float2 senderFacing;  // Absolute direction encoded by the signal angle relative to the front direction

    if (threadIdx.x == 0) {
        auto& cell = object->typeData.cell;
        auto& sender = cell.cellTypeData.communicator.modeData.sender;
        range = sender.range;
        senderPos = object->pos;
        senderFrontValid = cell.frontAngle != VALUE_NOT_SET_FLOAT;
        if (senderFrontValid) {
            auto encodedAngle = cell.neuralActivity.signals[Channels::CommunicatorAngle];
            senderFacing = ObjectConnectionProcessor::convertAngleSignalToAbsoluteDirection(data, object, encodedAngle);
        }
    }
    __syncthreads();

    // The sender must have an initialized front direction
    if (!senderFrontValid) {
        return;
    }

    // Matching lambda to check if a cell is a valid receiver
    auto isMatch = [&object](Object* otherObject) {
        // Must be a communicator in receiver mode
        if (otherObject->type != ObjectType_Cell) {
            return false;
        }
        if (otherObject->typeData.cell.cellType != CellType_Communicator
            || otherObject->typeData.cell.cellTypeData.communicator.mode != CommunicatorMode_Receiver) {
            return false;
        }

        // Must be from a different creature
        if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
            return false;
        }

        auto const& receiver = otherObject->typeData.cell.cellTypeData.communicator.modeData.receiver;

        // Check color restriction
        if (!((receiver.restrictToColors >> object->color) & 1)) {
            return false;
        }

        // Check lineage restriction
        if (receiver.restrictToLineage != LineageRestriction_No) {
            if (receiver.restrictToLineage == LineageRestriction_RelatedLineage) {
                if (!object->typeData.cell.creature->isSameLineage(otherObject->typeData.cell.creature)) {
                    return false;
                }
            } else if (receiver.restrictToLineage == LineageRestriction_UnrelatedLineage) {
                if (object->typeData.cell.creature->isSameLineage(otherObject->typeData.cell.creature)) {
                    return false;
                }
            }
        }

        return true;
    };

    // Calculate the total number of scan positions in the [-range, range] x [-range, range] region
    int diameter = 2 * range + 1;
    int totalPositions = diameter * diameter;

    // Each thread scans different positions in parallel
    for (int idx = threadIdx.x; idx < totalPositions; idx += blockDim.x) {
        int dx = (idx % diameter) - range;
        int dy = (idx / diameter) - range;

        // Check if within circular range
        float distSquared = static_cast<float>(dx * dx + dy * dy);
        if (distSquared > range * range) {
            continue;
        }

        float2 scanPos = {senderPos.x + static_cast<float>(dx), senderPos.y + static_cast<float>(dy)};
        data.objectMap.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto records = data.objectMap.getRecords();
        int otherIndex = data.objectMap.getFirstIndex(scanPos);
        while (otherIndex >= 0) {
            auto const& otherRecord = records[otherIndex];
            auto otherObject = otherRecord.self;
            if (isMatch(otherObject)) {
                tryTransmitSignal(data, object, otherObject, senderFacing);
            }
            otherIndex = otherRecord.nextObjectIndex;
        }
    }
}

__inline__ __device__ bool
CommunicatorProcessor::tryTransmitSignal(SimulationData& data, Object* senderObject, Object* receiverObject, float2 const& senderFacing)
{
    auto const& sender = senderObject->typeData.cell.cellTypeData.communicator.modeData.sender;
    if (sender.oneway) {
        // Direction gating: only send if the receiver lies in the half-plane opposite to the encoded facing direction
        auto toReceiver = data.objectMap.getCorrectedDirection(receiverObject->pos - senderObject->pos);
        if (Math::dot(toReceiver, senderFacing) > 0) {
            return false;
        }
    }

    // The receiver must have an initialized front direction
    if (receiverObject->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return false;
    }

    receiverObject->getLock();

    // Copy signal to receiver
    copyChannels(receiverObject->typeData.cell.neuralActivity.signals, senderObject->typeData.cell.neuralActivity.signals);

    // The encoded angle is relative to each cell's absolute front direction. senderFacing is the sender's absolute
    // encoded direction; convert it back into the receiver's frame so the absolute direction is preserved.
    auto receiverAngle = ObjectConnectionProcessor::convertAbsoluteDirectionToAngleSignal(data, receiverObject, senderFacing);
    receiverObject->typeData.cell.neuralActivity.signals[Channels::CommunicatorAngle] = receiverAngle;

    receiverObject->releaseLock();
    return true;
}
