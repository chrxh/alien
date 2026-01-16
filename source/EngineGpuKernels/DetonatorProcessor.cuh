#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entity.cuh"
#include "EnergyProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DetonatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DetonatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Detonator];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto& detonator = object->cellTypeData.detonator;
    if (SignalProcessor::isManuallyTriggered(data, object) && detonator.state == DetonatorState_Ready) {
        detonator.state = DetonatorState_Activated;
    }
    if (detonator.state == DetonatorState_Activated) {
        if (detonator.countdown >= 0) {
            --detonator.countdown;
        }
        if (detonator.countdown == -1) {
            object->event = CellEvent_Detonation;
            object->eventCounter = 10;
            detonator.countdown = 0;
            statistics.incNumDetonations(object->color);
            data.objectMap.executeForEach(object->pos, cudaSimulationParameters.detonatorRadius.value[object->color], object->detached, [&](Object* const& otherCell) {
                if (otherCell == object) {
                    return;
                }
                if (otherCell->fixed) {
                    return;
                }
                auto delta = data.objectMap.getCorrectedDirection(otherCell->pos - object->pos);
                auto lengthSquared = Math::lengthSquared(delta);
                if (lengthSquared > NEAR_ZERO) {
                    auto force = delta / lengthSquared * cudaSimulationParameters.detonatorRadius.value[object->color] * 2;
                    otherCell->vel += force;
                }
                if (otherCell->cellType == CellType_Detonator && otherCell->cellTypeData.detonator.state != DetonatorState_Exploded) {
                    if (data.primaryNumberGen.random() < cudaSimulationParameters.detonatorChainExplosionProbability.value[object->color]) {
                        otherCell->cellTypeData.detonator.state = DetonatorState_Activated;
                        otherCell->cellTypeData.detonator.countdown = 1;
                    }
                }
            });
            detonator.state = DetonatorState_Exploded;
        }
    }
}
