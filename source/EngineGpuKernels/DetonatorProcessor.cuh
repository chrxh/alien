#pragma once


#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "EngineInterface/CellFunctionConstants.h"
#include "Object.cuh"
#include "ParticleProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class DetonatorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void DetonatorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Detonator];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    auto& detonator = cell->cellFunctionData.detonator;
    if (activity.channels[0] >= abs(cudaSimulationParameters.cellFunctionDetonatorActivityThreshold) && detonator.state == DetonatorState_Ready) {
        detonator.state = DetonatorState_Activated;
    }
    if (detonator.state == DetonatorState_Activated) {
        if (detonator.countdown >= 0) {
            --detonator.countdown;
        }
        if (detonator.countdown == -1) {
            detonator.countdown = 0;
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.cellFunctionDetonatorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (otherCell->barrier) {
                        return;
                    }
                    auto delta = data.cellMap.getCorrectedDirection(otherCell->pos - cell->pos);
                    auto lengthSquared = Math::lengthSquared(delta);
                    if (lengthSquared > NEAR_ZERO) {
                        auto force = delta / lengthSquared * cudaSimulationParameters.cellFunctionDetonatorRadius[cell->color] * 2;
                        otherCell->vel += force;
                    }
                    if (otherCell->cellFunction == CellFunction_Detonator && otherCell->cellFunctionData.detonator.state != DetonatorState_Exploded) {
                        otherCell->cellFunctionData.detonator.state = DetonatorState_Activated;
                        otherCell->cellFunctionData.detonator.countdown = 1;
                    }
                });
            detonator.state = DetonatorState_Exploded;
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}
