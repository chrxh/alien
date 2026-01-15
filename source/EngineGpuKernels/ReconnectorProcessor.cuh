#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entity.cuh"
#include "EnergyParticleProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class ReconnectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell);

    __inline__ __device__ static void tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Object* cell);
    __inline__ __device__ static void removeConnections(SimulationData& data, SimulationStatistics& statistics, Object* cell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void ReconnectorProcessor::process(SimulationData& data, SimulationStatistics& result)
{
    auto& operations = data.cellTypeOperations[CellType_Reconnector];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, result, operations.at(i).object);
    }
}

__device__ __inline__ void ReconnectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    if (SignalProcessor::isManuallyTriggered(data, cell)) {
        if (object->signal.channels[Channels::CellTypeActivation] > 0) {
            tryCreateConnection(data, statistics, cell);
        } else {
            removeConnections(data, statistics, cell);
        }
    }
}

__inline__ __device__ void ReconnectorProcessor::tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    auto const& reconnector = object->cellTypeData.reconnector;
    auto const& reconnectorMode = reconnector.mode;

    Object* closestCell = nullptr;
    float closestDistance = 0;
    data.cellMap.executeForEach(object->pos, cudaSimulationParameters.reconnectorRadius.value[object->color], object->detached, [&](Object* const& otherCell) {

        // Skip if already connected or too closely connected
        if (ObjectConnectionProcessor::isConnectedConnected(cell, otherCell)) {
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
            if (object->isSameCreature(otherCell)) {
                return;
            }

            // Filter by color restriction
            auto const& restrictToColor = reconnector.modeData.reconnectCreature.restrictToColor;
            if (restrictToColor != 255 && otherCell->color != restrictToColor) {
                return;
            }

            // Filter by minimum number of cells in creature
            if (reconnector.modeData.reconnectCreature.minNumCells > 0
                && otherCell->creature->numObjects < reconnector.modeData.reconnectCreature.minNumCells) {
                return;
            }

            // Filter by maximum number of cells in creature
            if (reconnector.modeData.reconnectCreature.maxNumCells > 0
                && otherCell->creature->numObjects > reconnector.modeData.reconnectCreature.maxNumCells) {
                return;
            }

            // Filter by lineage restriction
            if (reconnector.modeData.reconnectCreature.restrictToLineage != LineageRestriction_No) {
                if (object->creature == nullptr) {
                    return;
                }
                if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_SameLineage) {
                    if (object->creature->lineageId != otherCell->creature->lineageId) {
                        return;
                    }
                } else if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_OtherLineage) {
                    if (object->creature->lineageId == otherCell->creature->lineageId) {
                        return;
                    }
                }
            }

        }

        // Check for own intersecting cells in between
        if (ObjectConnectionProcessor::existsOwnIntersectingCellInBetween(data, cell, otherCell)) {
            return;
        }

        auto distance = data.cellMap.getDistance(object->pos, otherCell->pos);
        if (!closestCell || distance < closestDistance) {
            closestCell = otherCell;
            closestDistance = distance;
        }
    });

    object->signal.channels[Channels::ReconnectorSuccess] = 0;
    if (closestCell) {
        SystemDoubleLock lock;
        lock.init(&object->locked, &closestCell->locked);
        if (lock.tryLock()) {
            if (object->numConnections < MAX_CELL_BONDS && closestCell->numConnections < MAX_CELL_BONDS) {
                ObjectConnectionProcessor::scheduleAddConnectionPair(data, cell, closestCell);
                object->signal.channels[Channels::ReconnectorSuccess] = 1;
                statistics.incNumReconnectorCreated(object->color);
            }
            lock.releaseLock();
        }
    }
}


__inline__ __device__ void ReconnectorProcessor::removeConnections(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    object->signal.channels[Channels::ReconnectorSuccess] = 0;

    if (!object->tryLock()) {
        return;
    }
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedCell = object->connections[i].object;
        bool shouldRemove = false;

        if (connectedCell->cellType == CellType_Structure || connectedCell->cellType == CellType_Free) {
            shouldRemove = true;
        }
        if (!object->isSameCreature(connectedCell)) {
            shouldRemove = true;
        }

        if (shouldRemove) {
            ObjectConnectionProcessor::scheduleDeleteConnectionPair(data, cell, connectedCell);
            object->signal.channels[Channels::ReconnectorSuccess] = 1;
            statistics.incNumReconnectorRemoved(object->color);
        }
    }
    object->releaseLock();
}
