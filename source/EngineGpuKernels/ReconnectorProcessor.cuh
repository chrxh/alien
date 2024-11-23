#pragma once


#include "CellConnectionProcessor.cuh"
#include "CellFunctionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "EngineInterface/CellFunctionConstants.h"
#include "Object.cuh"
#include "RadiationProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class ReconnectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal);
    __inline__ __device__ static void removeConnections(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ReconnectorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Reconnector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void ReconnectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto signal = CellFunctionProcessor::calcInputSignal(cell);
    CellFunctionProcessor::updateInvocationState(cell, signal);

    if (signal.channels[0] >= cudaSimulationParameters.cellFunctionReconnectorSignalThreshold) {
        tryCreateConnection(data, statistics, cell, signal);
    } else if (signal.channels[0] <= -cudaSimulationParameters.cellFunctionReconnectorSignalThreshold) {
        removeConnections(data, statistics, cell, signal);
    }
    CellFunctionProcessor::setSignal(cell, signal);
}

__inline__ __device__ void ReconnectorProcessor::tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal)
{
    auto const& reconnector = cell->cellFunctionData.reconnector;
    Cell* closestCell = nullptr;
    float closestDistance = 0;
    data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.cellFunctionReconnectorRadius[cell->color], cell->detached, [&](Cell* const& otherCell) {
        if (cell->creatureId != 0 && otherCell->creatureId == cell->creatureId) {
            return;
        }
        if (otherCell->barrier) {
            return;
        }
        if (reconnector.restrictToColor != 255 && otherCell->color != reconnector.restrictToColor) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToSameMutants && otherCell->mutationId != cell->mutationId) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToOtherMutants
            && (otherCell->mutationId == cell->mutationId || otherCell->mutationId == 0 || otherCell->mutationId == 1)) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToHandcraftedCells && otherCell->mutationId != 0) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToFreeCells && otherCell->mutationId != 1) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToLessComplexMutants
            && otherCell->genomeComplexity >= cell->genomeComplexity) {
            return;
        }
        if (reconnector.restrictToMutants == ReconnectorRestrictToMutants_RestrictToMoreComplexMutants
            && otherCell->genomeComplexity <= cell->genomeComplexity) {
            return;
        }
        if (CellConnectionProcessor::isConnectedConnected(cell, otherCell)) {
            return;
        }
        auto distance = data.cellMap.getDistance(cell->pos, otherCell->pos);
        if (!closestCell || distance < closestDistance) {
            closestCell = otherCell;
            closestDistance = distance;
        }
    });

    signal.channels[0] = 0;
    if (closestCell) {
        SystemDoubleLock lock;
        lock.init(&cell->locked, &closestCell->locked);
        if (lock.tryLock()) {
            if (cell->numConnections < MAX_CELL_BONDS && closestCell->numConnections < MAX_CELL_BONDS) {
                closestCell->maxConnections = min(max(closestCell->maxConnections, closestCell->numConnections + 1), MAX_CELL_BONDS);
                cell->maxConnections = min(max(cell->maxConnections, cell->numConnections + 1), MAX_CELL_BONDS);
                CellConnectionProcessor::scheduleAddConnectionPair(data, cell, closestCell);
                signal.channels[0] = 1;
                statistics.incNumReconnectorCreated(cell->color);
            }
            lock.releaseLock();
        }
    }
}

__inline__ __device__ void ReconnectorProcessor::removeConnections(SimulationData& data, SimulationStatistics& statistics, Cell* cell, Signal& signal)
{
    signal.channels[0] = 0;
    if (cell->tryLock()) {
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;
            if (connectedCell->creatureId != cell->creatureId) {
                CellConnectionProcessor::scheduleDeleteConnectionPair(data, cell, connectedCell);
                signal.channels[0] = 1;
                statistics.incNumReconnectorRemoved(cell->color);
            }
        }
        cell->releaseLock();
    }
}
