#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ObjectConnectionProcessor.cuh"
#include "ConstantMemory.cuh"
#include "Entity.cuh"
#include "EnergyProcessor.cuh"
#include "SignalProcessor.cuh"
#include "SimulationData.cuh"
#include "SimulationStatistics.cuh"

class ReconnectorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object);

    __inline__ __device__ static void tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Object* object);
    __inline__ __device__ static void removeConnections(SimulationData& data, SimulationStatistics& statistics, Object* object);
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

__device__ __inline__ void ReconnectorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    if (SignalProcessor::isManuallyTriggered(data, object)) {
        if (object->typeData.cell.signal.channels[Channels::CellTypeActivation] > 0) {
            tryCreateConnection(data, statistics, object);
        } else {
            removeConnections(data, statistics, object);
        }
    }
}

__inline__ __device__ void ReconnectorProcessor::tryCreateConnection(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    auto const& reconnector = object->typeData.cell.cellTypeData.reconnector;
    auto const& reconnectorMode = reconnector.mode;

    Object* closestCell = nullptr;
    float closestDistance = 0;
    data.objectMap.executeForEach(object->pos, cudaSimulationParameters.reconnectorRadius.value[object->color], object->detached, [&](Object* const& otherObject) {

        // Skip if already connected or too closely connected
        if (ObjectConnectionProcessor::isConnectedConnected(object, otherObject)) {
            return;
        }

        if (reconnectorMode == ReconnectorMode_Structure) {
            // Connect to structure cells only
            if (otherObject->type != ObjectType_Structure) {
                return;
            }
        } else if (reconnectorMode == ReconnectorMode_FreeCell) {
            // Connect to free cells only
            if (otherObject->type != ObjectType_FreeCell) {
                return;
            }

            // Filter by color restriction
            auto const& restrictToColor = reconnector.modeData.reconnectFreeCell.restrictToColor;
            if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                return;
            }
        } else if (reconnectorMode == ReconnectorMode_Creature) {
            // Connect to cells with creatures only
            if (otherObject->type != ObjectType_Cell) {
                return;
            }
            // Must be from a different creature
            if (otherObject->typeData.cell.creature == nullptr) {
                return;
            }
            if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                return;
            }

            // Filter by color restriction
            auto const& restrictToColor = reconnector.modeData.reconnectCreature.restrictToColor;
            if (restrictToColor != 255 && otherObject->color != restrictToColor) {
                return;
            }

            // Filter by minimum number of cells in creature
            if (reconnector.modeData.reconnectCreature.minNumCells > 0
                && otherObject->typeData.cell.creature->numObjects < reconnector.modeData.reconnectCreature.minNumCells) {
                return;
            }

            // Filter by maximum number of cells in creature
            if (reconnector.modeData.reconnectCreature.maxNumCells > 0
                && otherObject->typeData.cell.creature->numObjects > reconnector.modeData.reconnectCreature.maxNumCells) {
                return;
            }

            // Filter by lineage restriction
            if (reconnector.modeData.reconnectCreature.restrictToLineage != LineageRestriction_No) {
                if (object->typeData.cell.creature == nullptr) {
                    return;
                }
                if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_SameLineage) {
                    if (object->typeData.cell.creature->lineageId != otherObject->typeData.cell.creature->lineageId) {
                        return;
                    }
                } else if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_OtherLineage) {
                    if (object->typeData.cell.creature->lineageId == otherObject->typeData.cell.creature->lineageId) {
                        return;
                    }
                }
            }

        }

        // Check for own intersecting cells in between
        if (ObjectConnectionProcessor::existsOwnIntersectingObjectInBetween(data, object, otherObject)) {
            return;
        }

        auto distance = data.objectMap.getDistance(object->pos, otherObject->pos);
        if (!closestCell || distance < closestDistance) {
            closestCell = otherObject;
            closestDistance = distance;
        }
    });

    object->typeData.cell.signal.channels[Channels::ReconnectorSuccess] = 0;
    if (closestCell) {
        SystemDoubleLock lock;
        lock.init(&object->locked, &closestCell->locked);
        if (lock.tryLock()) {
            if (object->numConnections < MAX_OBJECT_CONNECTIONS && closestCell->numConnections < MAX_OBJECT_CONNECTIONS) {
                ObjectConnectionProcessor::scheduleAddConnectionPair(data, object, closestCell);
                object->typeData.cell.signal.channels[Channels::ReconnectorSuccess] = 1;
                statistics.incNumReconnectorCreated(object->color);
            }
            lock.releaseLock();
        }
    }
}


__inline__ __device__ void ReconnectorProcessor::removeConnections(SimulationData& data, SimulationStatistics& statistics, Object* object)
{
    object->typeData.cell.signal.channels[Channels::ReconnectorSuccess] = 0;

    if (!object->tryLock()) {
        return;
    }
    for (int i = 0; i < object->numConnections; ++i) {
        auto connectedObject = object->connections[i].object;
        bool shouldRemove = false;

        if (connectedObject->type == ObjectType_Structure || connectedObject->type == ObjectType_FreeCell) {
            shouldRemove = true;
        } else if (connectedObject->type == ObjectType_Cell) {
            if (!object->typeData.cell.isSameCreature(&connectedObject->typeData.cell)) {
                shouldRemove = true;
            }
        }

        if (shouldRemove) {
            ObjectConnectionProcessor::scheduleDeleteConnectionPair(data, object, connectedObject);
            object->typeData.cell.signal.channels[Channels::ReconnectorSuccess] = 1;
            statistics.incNumReconnectorRemoved(object->color);
        }
    }
    object->releaseLock();
}
