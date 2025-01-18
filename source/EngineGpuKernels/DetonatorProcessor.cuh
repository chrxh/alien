#pragma once


#include "CellConnectionProcessor.cuh"
#include "SignalProcessor.cuh"
#include "ConstantMemory.cuh"
#include "EngineInterface/CellTypeConstants.h"
#include "Object.cuh"
#include "RadiationProcessor.cuh"
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
    auto& operations = data.cellTypeOperations[CellType_Detonator];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void DetonatorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& detonator = cell->cellTypeData.detonator;
    if (abs(cell->signal.channels[0]) >= TRIGGER_THRESHOLD && detonator.state == DetonatorState_Ready) {
        detonator.state = DetonatorState_Activated;
    }
    if (detonator.state == DetonatorState_Activated) {
        if (detonator.countdown >= 0) {
            --detonator.countdown;
        }
        if (detonator.countdown == -1) {
            detonator.countdown = 0;
            statistics.incNumDetonations(cell->color);
            data.cellMap.executeForEach(
                cell->pos, cudaSimulationParameters.cellTypeDetonatorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
                    if (otherCell == cell) {
                        return;
                    }
                    if (otherCell->barrier) {
                        return;
                    }
                    auto delta = data.cellMap.getCorrectedDirection(otherCell->pos - cell->pos);
                    auto lengthSquared = Math::lengthSquared(delta);
                    if (lengthSquared > NEAR_ZERO) {
                        auto force = delta / lengthSquared * cudaSimulationParameters.cellTypeDetonatorRadius[cell->color] * 2;
                        otherCell->vel += force;
                    }
                    if (otherCell->cellType == CellType_Detonator && otherCell->cellTypeData.detonator.state != DetonatorState_Exploded) {
                        if (data.numberGen1.random() < cudaSimulationParameters.cellTypeDetonatorChainExplosionProbability[cell->color]) {
                            otherCell->cellTypeData.detonator.state = DetonatorState_Activated;
                            otherCell->cellTypeData.detonator.countdown = 1;
                        }
                    }
                });
            detonator.state = DetonatorState_Exploded;
        }
    }
}
