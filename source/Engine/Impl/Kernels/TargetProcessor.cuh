#pragma once

#include <Data/CellTypeConstants.h>

#include "SimulationData.cuh"

// Manages the targets that the sensor and receiver cells of a creature remember
class TargetProcessor
{
public:
    __inline__ __device__ static void store(SimulationData& data, Object* object, Target const& target);
    __inline__ __device__ static bool tryTakeOver(SimulationData& data, Object* object, Target const& target);

    // Must be called by all threads of a block and not while targets are changed
    __inline__ __device__ static bool findYoungest_block(SimulationData& data, Object* object, Target& result);

    // Between 0 and 1, 0 = expired
    __inline__ __device__ static float calcFreshness(SimulationData& data, Target const& target);

private:
    __inline__ __device__ static void storeWithoutLock(SimulationData& data, Object* object, Target const& target);
    __inline__ __device__ static uint16_t getCellIdPart(Object* object);
    __inline__ __device__ static uint32_t calcAge(SimulationData& data, Target const& target);

    static float constexpr NearbyCellRadius = 8.0f;
    static uint32_t constexpr NoTarget = 0xffffffff;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void TargetProcessor::store(SimulationData& data, Object* object, Target const& target)
{
    auto creature = object->typeData.cell.creature;
    creature->getLock();
    storeWithoutLock(data, object, target);
    creature->releaseLock();
}

// Takes over the target if it is not older than the one that the cell remembers
__inline__ __device__ bool TargetProcessor::tryTakeOver(SimulationData& data, Object* object, Target const& target)
{
    auto creature = object->typeData.cell.creature;
    creature->getLock();

    auto rememberedTarget = creature->findTarget(getCellIdPart(object));
    auto isAccepted = rememberedTarget == nullptr || calcAge(data, target) <= calcAge(data, *rememberedTarget);
    if (isAccepted) {
        storeWithoutLock(data, object, target);
    }

    creature->releaseLock();
    return isAccepted;
}

// The sensors and receivers near the cell are preferred to the others of the creature
__inline__ __device__ bool TargetProcessor::findYoungest_block(SimulationData& data, Object* object, Target& result)
{
    __shared__ uint32_t youngestKey;  // Consists of the age and the cell id part
    if (threadIdx.x == 0) {
        youngestKey = NoTarget;
    }
    __syncthreads();

    auto creature = object->typeData.cell.creature;
    auto considerTarget = [&](Target const& target) {
        if (calcFreshness(data, target) > 0.0f) {
            atomicMin(&youngestKey, calcAge(data, target) << 16 | target.cellIdPart);
        }
    };

    data.objectGrid.executeForEach_block(object->pos, NearbyCellRadius, object->detached(), [&](Object* const& otherObject) {
        if (otherObject->type != ObjectType_Cell || !object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
            return;
        }
        auto cellType = otherObject->typeData.cell.cellType;
        if (cellType != CellType_Sensor && cellType != CellType_Communicator) {
            return;
        }
        if (auto target = creature->findTarget(getCellIdPart(otherObject))) {
            considerTarget(*target);
        }
    });
    __syncthreads();

    auto isFoundNearby = youngestKey != NoTarget;
    __syncthreads();

    if (!isFoundNearby) {
        for (uint32_t i = threadIdx.x; i < creature->numTargets; i += blockDim.x) {
            considerTarget(creature->targets[i]);
        }
    }
    __syncthreads();

    if (youngestKey == NoTarget) {
        return false;
    }
    result = *creature->findTarget(youngestKey & 0xffff);
    return true;
}

__inline__ __device__ float TargetProcessor::calcFreshness(SimulationData& data, Target const& target)
{
    return max(0.0f, 1.0f - toFloat(calcAge(data, target)) / Const::CommunicatorTargetMaxAge);
}

// Replaces the own target, otherwise an expired one
__inline__ __device__ void TargetProcessor::storeWithoutLock(SimulationData& data, Object* object, Target const& target)
{
    auto creature = object->typeData.cell.creature;
    auto cellIdPart = getCellIdPart(object);

    auto entry = creature->findTarget(cellIdPart);
    for (uint32_t i = 0; i < creature->numTargets && entry == nullptr; ++i) {
        if (calcFreshness(data, creature->targets[i]) == 0.0f) {
            entry = &creature->targets[i];
        }
    }
    if (entry == nullptr) {
        data.entities.heap.growIfFull(creature->targets, creature->numTargets, creature->targetsCapacity);
        entry = &creature->targets[creature->numTargets];
        ++creature->numTargets;
    }
    *entry = target;
    entry->cellIdPart = cellIdPart;
}

__inline__ __device__ uint16_t TargetProcessor::getCellIdPart(Object* object)
{
    return static_cast<uint16_t>(object->id & 0xffff);
}

__inline__ __device__ uint32_t TargetProcessor::calcAge(SimulationData& data, Target const& target)
{
    return static_cast<uint32_t>(*data.timestep) - target.timestep;
}
