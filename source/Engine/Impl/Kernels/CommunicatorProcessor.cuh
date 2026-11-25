#pragma once

#include <Data/CellTypeConstants.h>
#include <Data/DistanceSignal.h>

#include "ObjectConnectionProcessor.cuh"
#include "SimulationStatistics.cuh"
#include "TargetProcessor.cuh"

class CommunicatorProcessor
{
public:
    __inline__ __device__ static void prepare(SimulationData& data);
    __inline__ __device__ static void selectTargets(SimulationData& data);
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void prepareReceiver(Object* object);

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void processSender(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void transmit(SimulationData& data, Object* senderObject, Object* receiverObject);
    __inline__ __device__ static void receiveTarget(SimulationData& data, Object* receiverObject, Target const& target);
    __inline__ __device__ static void writeTargetSignals(Object* receiverObject, bool isNewTarget);

    static int constexpr MaxReceivers = 512;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void CommunicatorProcessor::prepare(SimulationData& data)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto object = operations.at(i).object;
        object->typeData.cell.creature->hasCommunicators = true;
        if (object->typeData.cell.cellTypeData.communicator.mode == CommunicatorMode_Receiver) {
            prepareReceiver(object);
        }
    }
}

// The targets are selected before the transmissions, a target received in this cycle can thus be passed on in the next cycle at the earliest
__device__ __inline__ void CommunicatorProcessor::selectTargets(SimulationData& data)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto object = operations.at(i).object;
        auto& communicator = object->typeData.cell.cellTypeData.communicator;
        if (communicator.mode == CommunicatorMode_Sender && NeuronProcessor::isManuallyTriggered(data, object)) {
            Target target;
            auto hasTarget = TargetProcessor::findYoungest_block(data, object, target);
            if (threadIdx.x == 0) {
                communicator.modeData.sender.hasTarget = hasTarget;
                communicator.modeData.sender.target = target;
            }
        }
        __syncthreads();
    }
}

__device__ __inline__ void CommunicatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Communicator];
    auto partition = calcBlockPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).object);
    }
}

// Without a newly received target, the last one stays in the channels and only its freshness decreases
__device__ __inline__ void CommunicatorProcessor::prepareReceiver(Object* object)
{
    auto& receiver = object->typeData.cell.cellTypeData.communicator.modeData.receiver;
    receiver.nearestSenderDistance = Infinity<float>::value;
    receiver.targetFreshness = max(0.0f, receiver.targetFreshness - toFloat(TIMESTEPS_PER_CELL_FUNCTION) / Const::CommunicatorTargetMaxAge);
    if (receiver.targetFreshness > 0.0f) {
        writeTargetSignals(object, false);
    }
}

__device__ __inline__ void CommunicatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ bool shouldProcess;
    if (threadIdx.x == 0) {
        shouldProcess = NeuronProcessor::isManuallyTriggered(data, object);
    }
    __syncthreads();

    if (shouldProcess) {
        auto const& mode = object->typeData.cell.cellTypeData.communicator.mode;
        if (mode == CommunicatorMode_Sender) {
            processSender(data, statistics, object);
        }
        // Receiver mode: signals are set by senders and prepareReceiver
    }

    // Thread 0 must not overwrite the shared data for the next cell before every thread has finished this one, otherwise warps run out of step
    __syncthreads();
}

__device__ __inline__ void CommunicatorProcessor::processSender(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    __shared__ int range;
    __shared__ float2 senderPos;
    __shared__ Object* receivers[MaxReceivers];
    __shared__ int numReceivers;

    if (threadIdx.x == 0) {
        auto& cell = object->typeData.cell;
        auto& sender = cell.cellTypeData.communicator.modeData.sender;
        range = sender.range;
        senderPos = object->pos;
        numReceivers = 0;
    }
    __syncthreads();

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
        data.world.correctPosition(scanPos);

        // Check all cells at this position (including overlapping cells)
        auto records = data.objectGrid.getRecords();
        int otherIndex = data.objectGrid.getFirstIndex(scanPos);
        while (otherIndex >= 0) {
            auto const& otherRecord = records[otherIndex];
            auto otherObject = otherRecord.self;
            if (isMatch(otherObject)) {
                auto index = atomicAdd(&numReceivers, 1);
                if (index < MaxReceivers) {
                    receivers[index] = otherObject;
                }
            }
            otherIndex = otherRecord.nextObjectIndex;
        }
    }
    __syncthreads();

    // Only one thread transmits, otherwise two senders could wait for the locks of each other's receivers on AMD GPUs
    if (threadIdx.x == 0) {
        for (int i = 0; i < min(numReceivers, MaxReceivers); ++i) {
            transmit(data, object, receivers[i]);
        }
    }
}

// The message is taken over from the nearest sender, the target from each sender whose target is not older than the own one
__inline__ __device__ void CommunicatorProcessor::transmit(SimulationData& data, Object* senderObject, Object* receiverObject)
{
    auto toReceiver = data.world.getCorrectedDirection(receiverObject->pos - senderObject->pos);

    auto const& sender = senderObject->typeData.cell.cellTypeData.communicator.modeData.sender;
    if (sender.oneway && sender.hasTarget) {
        // Direction gating: only send if the receiver deviates by more than the one-way angle from the direction of the target
        auto toTarget = data.world.getCorrectedDirection(sender.target.pos - senderObject->pos);
        auto maxCosine = cosf(Const::CommunicatorOnewayMinAngle * Const::DEG_TO_RAD);
        if (Math::dot(toReceiver, toTarget) > maxCosine * Math::length(toReceiver) * Math::length(toTarget)) {
            return;
        }
    }

    // The receiver must have an initialized front direction
    if (receiverObject->typeData.cell.frontAngle == VALUE_NOT_SET_FLOAT) {
        return;
    }

    receiverObject->getLock();

    auto& receiver = receiverObject->typeData.cell.cellTypeData.communicator.modeData.receiver;
    auto distance = Math::length(toReceiver);
    if (distance < receiver.nearestSenderDistance) {
        receiver.nearestSenderDistance = distance;
        for (int channel = Channels::CommunicatorMessage; channel < STANDARD_NEURONS_PER_CELL; ++channel) {
            receiverObject->typeData.cell.neuralActivity.signals[channel] = senderObject->typeData.cell.neuralActivity.signals[channel];
        }
    }
    auto isTargetAllowed = sender.hasTarget && (sender.target.searchedColors & ~receiver.restrictToColors) == 0;
    if (isTargetAllowed && TargetProcessor::tryTakeOver(data, receiverObject, sender.target)) {
        receiveTarget(data, receiverObject, sender.target);
    }

    receiverObject->releaseLock();
}

__inline__ __device__ void CommunicatorProcessor::receiveTarget(SimulationData& data, Object* receiverObject, Target const& target)
{
    auto& receiver = receiverObject->typeData.cell.cellTypeData.communicator.modeData.receiver;
    auto toTarget = data.world.getCorrectedDirection(target.pos - receiverObject->pos);
    receiver.targetAngleSignal = ObjectConnectionProcessor::convertAbsoluteDirectionToAngleSignal(data, receiverObject, toTarget);
    receiver.targetFreshness = TargetProcessor::calcFreshness(data, target);
    receiver.targetDistanceSignal = DistanceSignal::encode(Math::length(toTarget));
    writeTargetSignals(receiverObject, true);
}

__inline__ __device__ void CommunicatorProcessor::writeTargetSignals(Object* receiverObject, bool isNewTarget)
{
    auto const& receiver = receiverObject->typeData.cell.cellTypeData.communicator.modeData.receiver;
    auto& signals = receiverObject->typeData.cell.neuralActivity.signals;
    signals[Channels::CommunicatorTargetReceived] = isNewTarget ? 1.0f : 0.0f;
    signals[Channels::CommunicatorAngle] = receiver.targetAngleSignal;
    signals[Channels::CommunicatorFreshness] = receiver.targetFreshness;
    signals[Channels::CommunicatorDistance] = receiver.targetDistanceSignal;
}
