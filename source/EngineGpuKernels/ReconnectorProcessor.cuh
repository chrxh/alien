#pragma once


#include <EngineInterface/CellTypeConstants.h>

#include "ConstantMemory.cuh"
#include "EnergyProcessor.cuh"
#include "Entities.cuh"
#include "ObjectConnectionProcessor.cuh"
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
    if (NeuronProcessor::isManuallyTriggered(data, object)) {
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
    data.objectMap.executeForEach(
        object->pos, cudaSimulationParameters.reconnectorRadius.value[object->color], object->detached, [&](Object* const& otherObject) {
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
                auto const& restrictToColors = reconnector.modeData.reconnectFreeCell.restrictToColors;
                if (!((restrictToColors >> otherObject->color) & 1)) {
                    return;
                }
            } else if (reconnectorMode == ReconnectorMode_Creature) {
                // Connect to cells with creatures only
                if (otherObject->type != ObjectType_Cell) {
                    return;
                }
                // Must be from a different creature
                if (object->typeData.cell.isSameCreature(&otherObject->typeData.cell)) {
                    return;
                }

                // Filter by color restriction
                auto const& restrictToColors = reconnector.modeData.reconnectCreature.restrictToColors;
                if (!((restrictToColors >> otherObject->color) & 1)) {
                    return;
                }

                // Filter by minimum number of cells in creature
                if (reconnector.modeData.reconnectCreature.minNumCells > 0
                    && otherObject->typeData.cell.creature->numCells < reconnector.modeData.reconnectCreature.minNumCells) {
                    return;
                }

                // Filter by maximum number of cells in creature
                if (reconnector.modeData.reconnectCreature.maxNumCells > 0
                    && otherObject->typeData.cell.creature->numCells > reconnector.modeData.reconnectCreature.maxNumCells) {
                    return;
                }

                // Filter by lineage restriction
                if (reconnector.modeData.reconnectCreature.restrictToLineage != LineageRestriction_No) {
                    if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_RelatedLineage) {
                        if (!object->typeData.cell.creature->genome->isRelatedLineage(otherObject->typeData.cell.creature->genome)) {
                            return;
                        }
                    } else if (reconnector.modeData.reconnectCreature.restrictToLineage == LineageRestriction_UnrelatedLineage) {
                        if (object->typeData.cell.creature->genome->isRelatedLineage(otherObject->typeData.cell.creature->genome)) {
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

        if (connectedObject->type != ObjectType_Cell) {
            shouldRemove = true;
        } else {
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
