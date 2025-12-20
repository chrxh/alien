#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "CellConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Object.cuh"
#include "EnergyParticleProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class ReconnectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static void removeConnections(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ReconnectorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Reconnector];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, result, operations.at(i).cell);
    }
}

__device__ __inline__ void ReconnectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        if (cell->signal.channels[Channels::CellTypeActivation] > 0) {
            tryCreateConnection(data, statistics, cell);
        } else {
            removeConnections(data, statistics, cell);
        }
    }
}

__inline__ __device__ void ReconnectorProcessor::tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto const& reconnector = cell->cellTypeData.reconnector;
    auto const& reconnectorMode = reconnector.mode;

    Cell* closestCell = nullptr;
    float closestDistance = 0;
    data.cellMap.executeForEach(cell->pos, cudaSimulationParameters.reconnectorRadius.value[cell->color], cell->detached, [&](Cell* const& otherCell) {

        // Skip if already connected or too closely connected
        if (CellConnectionProcessor::isConnectedConnected(cell, otherCell)) {
            return;
        }

        if (reconnectorMode == ReconnectorMode_Structure) {
            // Connect to structure cells only
            if (otherCell->cellType != CellType_Structure) {
                return;
            }
        } else if (reconnectorMode == ReconnectorMode_FreeCell) {
            // Connect to free cells only
            if (otherCell->cellType != CellType_Free) {
                return;
            }

            // Filter by color restriction
            auto const& restrictToColor = reconnector.modeData.reconnectFreeCell.restrictToColor;
            if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                return;
            }
        } else if (reconnectorMode == ReconnectorMode_Creature) {
            // Must be from a different creature
            if (otherCell->creature == nullptr) {
                return;
            }
            if (cell->isSameCreature(otherCell)) {
                return;
            }

            // Filter by color restriction
            auto const& restrictToColor = reconnector.modeData.reconnectCreature.restrictToColor;
            if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                return;
            }

            // Filter by minimum number of cells in creature
            if (reconnector.modeData.reconnectCreature.minNumCells > 0
                && otherCell->creature->numCells < reconnector.modeData.reconnectCreature.minNumCells) {
                return;
            }

            // Filter by maximum number of cells in creature
            if (reconnector.modeData.reconnectCreature.maxNumCells > 0
                && otherCell->creature->numCells > reconnector.modeData.reconnectCreature.maxNumCells) {
                return;
            }

            // Filter by lineage restriction
            if (reconnector.modeData.reconnectCreature.restrictToLineage != LineageRestriction_No) {
                if (cell->creature == nullptr) {
                    return;
                }
                if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_SameLineage) {
                    if (cell->creature->lineageId != otherCell->creature->lineageId) {
                        return;
                    }
                } else if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_OtherLineage) {
                    if (cell->creature->lineageId == otherCell->creature->lineageId) {
                        return;
                    }
                }
            }

            // Check for own intersecting cells in between
            if (CellConnectionProcessor::existsOwnIntersectingCellInBetween(data, cell, otherCell)) {
                return;
            }
        }

        auto distance = data.cellMap.getDistance(cell->pos, otherCell->pos);
        if (!closestCell || distance < closestDistance) {
            closestCell = otherCell;
            closestDistance = distance;
        }
    });

    cell->signal.channels[Channels::ReconnectorSuccess] = 0;
    if (closestCell) {
        SystemDoubleLock lock;
        lock.init(&cell->locked, &closestCell->locked);
        if (lock.tryLock()) {
            if (cell->numConnections < MAX_CELL_BONDS && closestCell->numConnections < MAX_CELL_BONDS) {
                CellConnectionProcessor::scheduleAddConnectionPair(data, cell, closestCell);
                cell->signal.channels[Channels::ReconnectorSuccess] = 1;
                statistics.incNumReconnectorCreated(cell->color);
            }
            lock.releaseLock();
        }
    }
}


__inline__ __device__ void ReconnectorProcessor::removeConnections(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    cell->signal.channels[Channels::ReconnectorSuccess] = 0;

    if (cell->tryLock()) {
        return;
    }
    for (int i = 0; i < cell->numConnections; ++i) {
        auto connectedCell = cell->connections[i].cell;
        bool shouldRemove = false;

        if (connectedCell->cellType == CellType_Structure || connectedCell->cellType == CellType_Free) {
            shouldRemove = true;
        }
        if (!cell->isSameCreature(connectedCell)) {
            shouldRemove = true;
        }

        if (shouldRemove) {
            CellConnectionProcessor::scheduleDeleteConnectionPair(data, cell, connectedCell);
            cell->signal.channels[Channels::ReconnectorSuccess] = 1;
            statistics.incNumReconnectorRemoved(cell->color);
        }
    }
    cell->releaseLock();
}
