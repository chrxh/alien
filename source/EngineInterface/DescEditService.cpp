#include "DescEditService.h"

#include <cmath>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include <Base/Math.h>
#include <Base/Physics.h>

#include <EngineInterface/NumberGenerator.h>

#include "GenomeDesc.h"
#include "SpaceCalculator.h"

namespace
{
    IntVector2D calcNumTiles(IntVector2D const& origWorldSize, IntVector2D const& worldSize)
    {
        return {(worldSize.x + origWorldSize.x - 1) / origWorldSize.x, (worldSize.y + origWorldSize.y - 1) / origWorldSize.y};
    }

    bool isCoveredByTile(RealVector2D const& posInOrigWorld, IntVector2D const& tileOffset, IntVector2D const& worldSize)
    {
        return toFloat(tileOffset.x) + posInOrigWorld.x < toFloat(worldSize.x) && toFloat(tileOffset.y) + posInOrigWorld.y < toFloat(worldSize.y);
    }

    void dropConnectionsToRemovedObjects(ObjectDesc& object, std::unordered_set<uint64_t> const& removedObjectIds)
    {
        std::vector<ConnectionDesc> remainingConnections;
        auto angleToTransfer = 0.0f;
        for (auto const& connection : object._connections) {
            if (removedObjectIds.contains(connection._objectId)) {
                angleToTransfer += connection._angleFromPrevious;
            } else {
                remainingConnections.emplace_back(connection);
                remainingConnections.back()._angleFromPrevious += angleToTransfer;
                angleToTransfer = 0.0f;
            }
        }
        if (!remainingConnections.empty()) {
            remainingConnections.front()._angleFromPrevious += angleToTransfer;
        }
        object._connections = remainingConnections;
    }

    void dropReferencesToRemovedObjects(ContentDesc& content, std::unordered_set<uint64_t> const& removedObjectIds)
    {
        for (auto& object : content._objects) {
            dropConnectionsToRemovedObjects(object, removedObjectIds);

            if (object.getObjectType() != ObjectType_Cell) {
                continue;
            }
            auto& constructor = object.getCellRef()._constructor;
            if (constructor.has_value() && constructor->_lastConstructedCellId.has_value()
                && removedObjectIds.contains(constructor->_lastConstructedCellId.value())) {
                constructor->_lastConstructedCellId.reset();
            }
        }
    }

    std::unordered_map<uint64_t, int> calcNumCellsByCreatureId(ContentDesc const& content)
    {
        std::unordered_map<uint64_t, int> result;
        for (auto const& object : content._objects) {
            if (object.getObjectType() == ObjectType_Cell) {
                ++result[object.getCellRef()._creatureId];
            }
        }
        return result;
    }

    void dropUnusedGenomes(ContentDesc& content, std::unordered_set<uint64_t> const& genomeIdsToCheck)
    {
        std::unordered_set<uint64_t> usedGenomeIds;
        for (auto const& creature : content._creatures) {
            usedGenomeIds.insert(creature._genomeId);
        }
        std::erase_if(content._genomes, [&](auto const& genome) { return genomeIdsToCheck.contains(genome._id) && !usedGenomeIds.contains(genome._id); });
    }

    void syncCreaturesWithRemainingCells(ContentDesc& content)
    {
        auto numCellsByCreatureId = calcNumCellsByCreatureId(content);

        std::unordered_set<uint64_t> removedCreatureIds;
        std::unordered_set<uint64_t> genomeIdsOfRemovedCreatures;
        for (auto const& creature : content._creatures) {
            if (!numCellsByCreatureId.contains(creature._id)) {
                removedCreatureIds.insert(creature._id);
                genomeIdsOfRemovedCreatures.insert(creature._genomeId);
            }
        }
        std::erase_if(content._creatures, [&](auto const& creature) { return removedCreatureIds.contains(creature._id); });

        for (auto& creature : content._creatures) {
            creature._numCells = numCellsByCreatureId.at(creature._id);
            if (creature._ancestorId.has_value() && removedCreatureIds.contains(creature._ancestorId.value())) {
                creature._ancestorId.reset();
            }
        }
        dropUnusedGenomes(content, genomeIdsOfRemovedCreatures);
    }

    ContentDesc cutOutTile(ContentDesc const& flattenedContent, IntVector2D const& origWorldSize, IntVector2D const& tileOffset, IntVector2D const& worldSize)
    {
        SpaceCalculator origSpace(origWorldSize);
        auto isCovered = [&](RealVector2D const& pos) { return isCoveredByTile(origSpace.getCorrectedPosition(pos), tileOffset, worldSize); };

        auto result = flattenedContent;

        std::unordered_set<uint64_t> removedObjectIds;
        for (auto const& object : result._objects) {
            if (!isCovered(object._pos)) {
                removedObjectIds.insert(object._id);
            }
        }
        std::erase_if(result._objects, [&](auto const& object) { return removedObjectIds.contains(object._id); });
        std::erase_if(result._energies, [&](auto const& energy) { return !isCovered(energy._pos); });

        dropReferencesToRemovedObjects(result, removedObjectIds);
        syncCreaturesWithRemainingCells(result);
        return result;
    }

    void reserveExistingIds(ContentDesc const& content)
    {
        auto& numberGenerator = NumberGenerator::get();
        for (auto const& object : content._objects) {
            numberGenerator.adaptMaxEntityId(object._id);
        }
        for (auto const& energy : content._energies) {
            numberGenerator.adaptMaxEntityId(energy._id);
        }
        for (auto const& creature : content._creatures) {
            numberGenerator.adaptMaxEntityId(creature._id);
        }
        for (auto const& genome : content._genomes) {
            numberGenerator.adaptMaxEntityId(genome._id);
        }
    }

    void mapPositionsIntoWorld(ContentDesc& content, IntVector2D const& worldSize)
    {
        SpaceCalculator space(worldSize);
        for (auto& object : content._objects) {
            object._pos = space.getCorrectedPosition(object._pos);
        }
        for (auto& energy : content._energies) {
            energy._pos = space.getCorrectedPosition(energy._pos);
        }
    }
}

void DescEditService::scaleContent(ContentDesc& description, IntVector2D const& origWorldSize, IntVector2D const& worldSize) const
{
    if (origWorldSize.x <= 0 || origWorldSize.y <= 0 || worldSize.x <= 0 || worldSize.y <= 0 || origWorldSize == worldSize) {
        return;
    }

    reserveExistingIds(description);
    flattenTopology(description, origWorldSize);

    auto numTiles = calcNumTiles(origWorldSize, worldSize);

    ContentDesc result;
    for (auto tileX : std::views::iota(0, numTiles.x)) {
        for (auto tileY : std::views::iota(0, numTiles.y)) {
            IntVector2D tileOffset{tileX * origWorldSize.x, tileY * origWorldSize.y};
            auto tile = cutOutTile(description, origWorldSize, tileOffset, worldSize);
            shift(tile, toRealVector2D(tileOffset));

            auto isDuplicate = tileOffset != IntVector2D{0, 0};
            result.add(std::move(tile), isDuplicate);
        }
    }
    mapPositionsIntoWorld(result, worldSize);

    description = result;
}

namespace
{
    std::vector<int> getObjectIndicesWithinRadius(
        ContentDesc const& description,
        std::unordered_map<int, std::unordered_map<int, std::vector<int>>> const& objectIndicesBySlot,
        RealVector2D const& pos,
        float radius)
    {
        std::vector<int> result;
        IntVector2D upperLeftIntPos{toInt(pos.x - radius - 0.5f), toInt(pos.y - radius - 0.5f)};
        IntVector2D lowerRightIntPos{toInt(pos.x + radius + 0.5f), toInt(pos.y + radius + 0.5f)};
        for (int x = upperLeftIntPos.x; x <= lowerRightIntPos.x; ++x) {
            for (int y = upperLeftIntPos.y; y <= lowerRightIntPos.y; ++y) {
                if (objectIndicesBySlot.find(x) != objectIndicesBySlot.end()) {
                    if (objectIndicesBySlot.at(x).find(y) != objectIndicesBySlot.at(x).end()) {
                        for (auto const& objectIndex : objectIndicesBySlot.at(x).at(y)) {
                            auto const& object = description._objects.at(objectIndex);
                            if (Math::length(object._pos - pos) <= radius) {
                                result.emplace_back(objectIndex);
                            }
                        }
                    }
                }
            }
        }
        std::sort(result.begin(), result.end(), [&](int index1, int index2) {
            auto const& object1 = description._objects.at(index1);
            auto const& object2 = description._objects.at(index2);
            return Math::length(object1._pos - pos) < Math::length(object2._pos - pos);
        });
        return result;
    }
}

void DescEditService::addIfSpaceAvailable(ContentDesc& result, Occupancy& occupancy, ContentDesc const& toAdd, float distance, IntVector2D const& worldSize)
    const
{
    SpaceCalculator space(worldSize);

    for (auto const& object : toAdd._objects) {
        if (!isCellPresent(occupancy, space, object._pos, distance)) {
            result._objects.emplace_back(object);
            occupancy[toIntVector2D(object._pos)].emplace_back(object._pos);
        }
    }
    for (auto const& energy : toAdd._energies) {
        if (!isCellPresent(occupancy, space, energy._pos, distance)) {
            result._energies.emplace_back(energy);
            occupancy[toIntVector2D(energy._pos)].emplace_back(energy._pos);
        }
    }
}

void DescEditService::flattenTopology(ContentDesc& description, IntVector2D const& worldSize) const
{
    SpaceCalculator space(worldSize);
    auto cache = description.createCache();

    std::unordered_set<uint64_t> workingCellIds;
    std::unordered_set<uint64_t> freeCellIds;

    for (auto const& object : description._objects) {
        freeCellIds.insert(object._id);
    }
    while (!workingCellIds.empty() || !freeCellIds.empty()) {

        // Take an arbitrary cell to start with
        if (workingCellIds.empty()) {
            workingCellIds.insert(*freeCellIds.begin());
            freeCellIds.erase(freeCellIds.begin());
        }

        // Process working cells: find connected free cells and correct topology
        std::unordered_set<uint64_t> newWorkingCellIds;
        for (auto const& objectId : workingCellIds) {
            auto& object = description.getObjectRef(objectId, cache);

            for (auto const& connection : object._connections) {
                if (freeCellIds.contains(connection._objectId)) {
                    // Do topology correction
                    auto& otherObject = description.getObjectRef(connection._objectId, cache);
                    otherObject._pos += space.getCorrectionIncrement(object._pos, otherObject._pos);

                    freeCellIds.erase(connection._objectId);
                    newWorkingCellIds.insert(connection._objectId);
                }
            }
        }
        workingCellIds = newWorkingCellIds;
    }
}

void DescEditService::reconnectObjects(ContentDesc& description, float maxDistance) const
{
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> objectIndicesBySlot;

    int index = 0;
    for (auto& object : description._objects) {
        object._connections.clear();
        objectIndicesBySlot[toInt(object._pos.x)][toInt(object._pos.y)].emplace_back(toInt(index));
        ++index;
    }

    auto cache = description.createCache();
    auto existsCrossingConnection = [&](ObjectDesc const& object, ObjectDesc const& nearbyObject) {
        for (auto const& connection : object._connections) {
            auto const& connectedObject = description.getObjectRef(connection._objectId, cache);
            for (auto const& otherConnection : connectedObject._connections) {
                auto const& otherConnectedObject = description.getObjectRef(otherConnection._objectId, cache);
                if (Math::isCrossing(object._pos, nearbyObject._pos, connectedObject._pos, otherConnectedObject._pos)) {
                    return true;
                }
            }
        }
        for (auto const& connection : nearbyObject._connections) {
            auto const& connectedObject = description.getObjectRef(connection._objectId, cache);
            for (auto const& otherConnection : connectedObject._connections) {
                auto const& otherConnectedObject = description.getObjectRef(otherConnection._objectId, cache);
                if (Math::isCrossing(object._pos, nearbyObject._pos, connectedObject._pos, otherConnectedObject._pos)) {
                    return true;
                }
            }
        }
        return false;
    };

    for (auto& object : description._objects) {
        if (object.getObjectType() == ObjectType_Fluid) {
            continue;
        }
        auto nearbyObjectIndices = getObjectIndicesWithinRadius(description, objectIndicesBySlot, object._pos, maxDistance);
        for (auto const& nearbyObjectIndex : nearbyObjectIndices) {
            auto const& nearbyObject = description._objects.at(nearbyObjectIndex);
            if (nearbyObject.getObjectType() == ObjectType_Fluid) {
                continue;
            }
            if (object._id != nearbyObject._id && object._connections.size() < MAX_OBJECT_CONNECTIONS
                && nearbyObject._connections.size() < MAX_OBJECT_CONNECTIONS && !object.isConnectedTo(nearbyObject._id)
                && !existsCrossingConnection(object, nearbyObject)) {
                description.addConnection(object._id, nearbyObject._id, cache);
            }
        }
    }
}

void DescEditService::setCenter(ContentDesc& description, RealVector2D const& center) const
{
    auto origCenter = calcCenter(description);
    auto delta = center - origCenter;
    shift(description, delta);
}

RealVector2D DescEditService::calcCenter(ContentDesc const& description) const
{
    RealVector2D result;
    auto numEntities = description._objects.size() + description._energies.size();
    for (auto const& object : description._objects) {
        result += object._pos;
    }

    for (auto const& energyParticle : description._energies) {
        result += energyParticle._pos;
    }
    result /= numEntities;
    return result;
}

void DescEditService::shift(ContentDesc& description, RealVector2D const& delta) const
{
    for (auto& object : description._objects) {
        object._pos += delta;
    }

    for (auto& energyParticle : description._energies) {
        energyParticle._pos += delta;
    }
}

void DescEditService::rotate(ContentDesc& description, float angle) const
{
    auto rotationMatrix = Math::calcRotationMatrix(angle);
    auto center = calcCenter(description);

    auto rotate = [&](RealVector2D& pos) {
        auto relPos = pos - center;
        auto rotatedRelPos = rotationMatrix * relPos;
        pos = center + rotatedRelPos;
    };
    for (auto& object : description._objects) {
        rotate(object._pos);
    }
    for (auto& energyParticle : description._energies) {
        rotate(energyParticle._pos);
    }
}

void DescEditService::accelerate(ContentDesc& description, RealVector2D const& velDelta, float angularVelDelta) const
{
    auto center = calcCenter(description);

    auto accelerate = [&](RealVector2D const& pos, RealVector2D& vel) {
        auto relPos = pos - center;
        vel += Physics::tangentialVelocity(relPos, velDelta, angularVelDelta);
    };
    for (auto& object : description._objects) {
        accelerate(object._pos, object._vel);
    }
    for (auto& energyParticle : description._energies) {
        accelerate(energyParticle._pos, energyParticle._vel);
    }
}

void DescEditService::removeCell(ContentDesc& description, uint64_t objectId) const
{
    std::erase_if(description._objects, [&](auto const& object) { return object._id == objectId; });

    // Check if any creatures have no cells left
    std::unordered_set<uint64_t> creaturesWithCells;
    for (auto const& object : description._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            creaturesWithCells.insert(object.getCellRef()._creatureId);
        }
    }
    std::erase_if(description._creatures, [&](auto const& creature) { return !creaturesWithCells.contains(creature._id); });

    // Check if any genomes have no creatures left
    std::unordered_set<uint64_t> genomesWithCreatures;
    for (auto const& creature : description._creatures) {
        genomesWithCreatures.insert(creature._genomeId);
    }
    std::erase_if(description._genomes, [&](auto const& genome) { return !genomesWithCreatures.contains(genome._id); });

    // Adapt connections
    for (auto& object : description._objects) {
        for (int i = 0, numConnections = object._connections.size(); i < numConnections; ++i) {
            auto const& connection = object._connections[i];
            if (connection._objectId == objectId) {
                auto angleToAdd = connection._angleFromPrevious;
                for (int k = i; k < numConnections - 1; ++k) {
                    object._connections.at(k) = object._connections.at(k + 1);
                }

                if (i < numConnections - 1) {
                    object._connections.at(i)._angleFromPrevious += angleToAdd;
                } else {
                    object._connections.at(0)._angleFromPrevious += angleToAdd;
                }

                object._connections.pop_back();
                return;
            }
        }
    }
}

void DescEditService::removeCellIf(ContentDesc& description, std::function<bool(ObjectDesc const&)> const& predicate) const
{
    std::unordered_set<uint64_t> removedCellIds;
    auto extPredicate = [&](ObjectDesc const& object) {
        auto result = predicate(object);
        if (result) {
            removedCellIds.insert(object._id);
        }
        return result;
    };

    std::erase_if(description._objects, extPredicate);

    // Check if any creatures have no cells left
    std::unordered_set<uint64_t> creaturesWithCells;
    for (auto const& object : description._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            creaturesWithCells.insert(object.getCellRef()._creatureId);
        }
    }
    std::erase_if(description._creatures, [&](auto const& creature) { return !creaturesWithCells.contains(creature._id); });

    for (auto& object : description._objects) {
        for (int i = 0, numConnections = object._connections.size(); i < numConnections; ++i) {
            auto const& connection = object._connections[i];
            if (removedCellIds.contains(connection._objectId)) {
                auto angleToAdd = connection._angleFromPrevious;
                for (int k = i; k < numConnections - 1; ++k) {
                    object._connections.at(k) = object._connections.at(k + 1);
                }

                if (i < numConnections - 1) {
                    object._connections.at(i)._angleFromPrevious += angleToAdd;
                } else {
                    object._connections.at(0)._angleFromPrevious += angleToAdd;
                }

                object._connections.pop_back();
                return;
            }
        }
    }
}

bool DescEditService::isCellPresent(Occupancy const& cellPosBySlot, SpaceCalculator const& spaceCalculator, RealVector2D const& posToCheck, float distance)
    const
{
    auto intPos = toIntVector2D(posToCheck);

    auto getMatchingSlots = [&cellPosBySlot](IntVector2D const& intPos) {
        auto findResult = cellPosBySlot.find(intPos);
        if (findResult != cellPosBySlot.end()) {
            return findResult->second;
        }
        return std::vector<RealVector2D>{};
    };

    auto isOccupied = [&](std::vector<RealVector2D> const& cellPositions) {
        for (auto const& cellPos : cellPositions) {
            auto otherPos = spaceCalculator.getCorrectedPosition(cellPos);
            if (Math::length(posToCheck - otherPos) < distance) {
                return true;
            }
        }
        return false;
    };

    auto distanceInt = toInt(ceilf(distance));
    for (int dx = -distanceInt; dx <= distanceInt; ++dx) {
        for (int dy = -distanceInt; dy <= distanceInt; ++dy) {
            if (isOccupied(getMatchingSlots({intPos.x + dx, intPos.y + dy}))) {
                return true;
            }
        }
    }
    return false;
}

uint64_t DescEditService::getId(ExtendedObjectOrEnergyDesc const& entity) const
{
    if (std::holds_alternative<ExtendedObjectDesc>(entity)) {
        return std::get<ExtendedObjectDesc>(entity).object._id;
    }
    return std::get<EnergyDesc>(entity)._id;
}

RealVector2D DescEditService::getPos(ExtendedObjectOrEnergyDesc const& entity) const
{
    if (std::holds_alternative<ExtendedObjectDesc>(entity)) {
        return std::get<ExtendedObjectDesc>(entity).object._pos;
    }
    return std::get<EnergyDesc>(entity)._pos;
}

std::vector<ExtendedObjectOrEnergyDesc> DescEditService::getObjects(ContentDesc const& description) const
{
    std::vector<ExtendedObjectOrEnergyDesc> result;
    for (auto const& energyParticle : description._energies) {
        result.emplace_back(energyParticle);
    }

    // Build a map of creatureId to genome
    std::unordered_map<uint64_t, GenomeDesc> genomeByCreatureId;
    for (auto const& creature : description._creatures) {
        auto genomeIt =
            std::find_if(description._genomes.begin(), description._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
        if (genomeIt != description._genomes.end()) {
            genomeByCreatureId.emplace(creature._id, *genomeIt);
        }
    }
    auto cache = description.createCache();

    for (auto const& object : description._objects) {
        ExtendedObjectDesc extObject;
        extObject.object = object;
        if (object.getObjectType() == ObjectType_Cell) {
            auto const& cell = object.getCellRef();
            extObject.creature = description.getCreatureRef(cell._creatureId, cache);
            auto genomeIt = genomeByCreatureId.find(cell._creatureId);
            if (genomeIt != genomeByCreatureId.end()) {
                extObject.genome = genomeIt->second;
            }
        }
        result.emplace_back(extObject);
    }
    return result;
}
