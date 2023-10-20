#pragma once


#include "EngineInterface/CellFunctionConstants.h"

#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Object.cuh"
#include "ParticleProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"
#include "CellConnectionProcessor.cuh"

class ReconectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ReconectorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Reconnector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void ReconectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (abs(activity.channels[0]) >= 0.1f) {

        Cell* closestCell = nullptr;
        float closestDistance = 0;
        data.cellMap.executeForEach(cell->pos, 2.0f, cell->detached, [&](auto const& otherCell) {
            if (cell->creatureId != 0 && otherCell->creatureId == cell->creatureId) {
                return;
            }
            if (CellConnectionProcessor::isConnectedConnected(cell, otherCell)) {
                return;
            }
            if (otherCell->barrier) {
                return;
            }
            auto distance = data.cellMap.getDistance(cell->pos, otherCell->pos);
            if (!closestCell || distance < closestDistance) {
                closestCell = otherCell;
                closestDistance = distance;
            }
        });

        if (closestCell) {
            CellConnectionProcessor::scheduleAddConnectionPair(cell, closestCell);
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}

