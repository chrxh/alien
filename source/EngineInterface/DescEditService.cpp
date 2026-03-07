#include "DescEditService.h"

#include <cmath>
#include <unordered_map>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include <Base/Math.h>
#include <Base/Physics.h>

#include <EngineInterface/NumberGenerator.h>

#include "GenomeDesc.h"
#include "SpaceCalculator.h"

Desc DescEditService::createRect(CreateRectParameters const& parameters) const
{
    Desc result;
    for (int i = 0; i < parameters._width; ++i) {
        for (int j = 0; j < parameters._height; ++j) {
            result._objects.emplace_back(ObjectDesc()
                                             .pos({toFloat(i) * parameters._cellDistance, toFloat(j) * parameters._cellDistance})
                                             .stiffness(parameters._stiffness)
                                             .color(parameters._color)
                                             .fixed(parameters._fixed)
                                             .sticky(parameters._sticky)
                                             .type(parameters._objectType));
        }
    }
    reconnectCells(result, parameters._cellDistance * 1.1f);
    setCenter(result, parameters._center);
    return result;
}

Desc DescEditService::createHex(CreateHexParameters const& parameters) const
{
    Desc result;
    auto incY = sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < parameters._layers; ++j) {
        for (int i = -(parameters._layers - 1); i < parameters._layers - j; ++i) {

            //create cell: upper layer
            result._objects.emplace_back(ObjectDesc()
                                             .stiffness(parameters._stiffness)
                                             .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(-j * incY)})
                                             .color(parameters._color)
                                             .fixed(parameters._fixed)
                                             .sticky(parameters._sticky)
                                             .type(parameters._objectType));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                result._objects.emplace_back(ObjectDesc()
                                                 .stiffness(parameters._stiffness)
                                                 .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(j * incY)})
                                                 .color(parameters._color)
                                                 .fixed(parameters._fixed)
                                                 .type(parameters._objectType));
            }
        }
    }

    reconnectCells(result, parameters._cellDistance * 1.5f);
    setCenter(result, parameters._center);

    return result;
}

Desc DescEditService::createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters) const
{
    Desc result;

    if (parameters._radius <= 1 + NEAR_ZERO) {
        result._objects.emplace_back(ObjectDesc()
                                         .pos(parameters._center)
                                         .stiffness(parameters._stiffness)
                                         .color(parameters._color)
                                         .fixed(parameters._fixed)
                                         .sticky(parameters._sticky)
                                         .type(StructureDesc()));
        return result;
    }

    auto centerRow = toInt(parameters._center.y / parameters._cellDistance);
    auto radiusRow = toInt(parameters._radius / parameters._cellDistance);

    auto startYRow = centerRow - radiusRow;
    auto radiusRounded = radiusRow * parameters._cellDistance;
    for (float dx = -radiusRounded; dx <= radiusRounded + NEAR_ZERO; dx += parameters._cellDistance) {
        int row = 0;
        for (float dy = -radiusRounded; dy <= radiusRounded + NEAR_ZERO; dy += parameters._cellDistance, ++row) {
            float evenRowIncrement = (startYRow + row) % 2 == 0 ? parameters._cellDistance / 2 : 0.0f;
            auto dxMod = dx + evenRowIncrement;
            if (dxMod * dxMod + dy * dy > radiusRounded * radiusRounded + NEAR_ZERO) {
                continue;
            }
            result._objects.emplace_back(ObjectDesc()
                                             .stiffness(parameters._stiffness)
                                             .pos({parameters._center.x + dxMod, parameters._center.y + dy})
                                             .color(parameters._color)
                                             .fixed(parameters._fixed)
                                             .sticky(parameters._sticky)
                                             .type(StructureDesc()));
        }
    }
    return result;
}

namespace
{
    void topologyCorrection(SpaceCalculator const& spaceCalc, Desc& description, uint64_t creatureId)
    {
        ObjectDesc* refCell = nullptr;
        for (auto& object : description._objects) {
            if (object.getObjectType() != ObjectType_Cell) {
                continue;
            }
            if (object.getCellRef()._creatureId == creatureId) {
                if (!refCell) {
                    refCell = &object;
                }
                auto topologyCorrection = spaceCalc.getCorrectionIncrement(refCell->_pos, object._pos);
                object._pos = object._pos + topologyCorrection;
            }
        }
    }

    void correctConnectionsForNonCreatures(Desc& description, IntVector2D const& worldSize)
    {
        auto threshold = std::min(worldSize.x, worldSize.y) / 3;
        auto cache = description.createCache();

        for (auto& object : description._objects) {
            if (object.getObjectType() == ObjectType_Cell) {
                continue;
            }
            std::vector<ConnectionDesc> newConnections;
            float angleToAdd = 0;
            for (auto connection : object._connections) {
                auto const& connectedObject = description.getObjectRef(connection._objectId, cache);
                if (Math::length(object._pos - connectedObject._pos) > threshold) {
                    angleToAdd += connection._angleFromPrevious;
                } else {
                    connection._angleFromPrevious += angleToAdd;
                    angleToAdd = 0;
                    newConnections.emplace_back(connection);
                }
            }
            if (angleToAdd > NEAR_ZERO && !newConnections.empty()) {
                newConnections.front()._angleFromPrevious += angleToAdd;
            }
            object._connections = newConnections;
        }
    }

}

void DescEditService::duplicate(Desc& description, IntVector2D const& origSize, IntVector2D const& size) const
{
    correctConnectionsForNonCreatures(description, origSize);

    SpaceCalculator spaceCalc(origSize);

    Desc result;
    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            auto clone = Desc(description);
            clone.assignNewEntityIds();
            for (auto const& creature : clone._creatures) {
                topologyCorrection(spaceCalc, clone, creature._id);
                result._creatures.emplace_back(creature);
            }
            for (auto& object : clone._objects) {
                object._pos = RealVector2D{object._pos.x + incX, object._pos.y + incY};
                result._objects.emplace_back(object);
            }
            for (auto energyParticle : clone._energies) {
                auto origPos = energyParticle._pos;
                energyParticle._pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                result._energies.emplace_back(energyParticle);
            }
            result._genomes.insert(result._genomes.end(), clone._genomes.begin(), clone._genomes.end());
        }
    }

    description = Desc(result);
}

namespace
{
    std::vector<int> getObjectIndicesWithinRadius(
        Desc const& description,
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

Desc DescEditService::gridMultiply(Desc const& input, GridMultiplyParameters const& parameters) const
{
    Desc result;
    auto clone = input;
    auto cloneTemplate = input;
    for (int i = 0; i < parameters._horizontalNumber; ++i) {
        for (int j = 0; j < parameters._verticalNumber; ++j) {
            auto templateData = [&] {
                if (i == 0 && j == 0) {
                    return clone;
                }
                return cloneTemplate;
            }();
            shift(templateData, {i * parameters._horizontalDistance, j * parameters._verticalDistance});
            rotate(templateData, i * parameters._horizontalAngleInc + j * parameters._verticalAngleInc);
            accelerate(
                templateData,
                {i * parameters._horizontalVelXinc + j * parameters._verticalVelXinc, i * parameters._horizontalVelYinc + j * parameters._verticalVelYinc},
                i * parameters._horizontalAngularVelInc + j * parameters._verticalAngularVelInc);

            result.add(std::move(templateData));
        }
    }

    return result;
}

Desc DescEditService::randomMultiply(
    Desc const& input,
    RandomMultiplyParameters const& parameters,
    IntVector2D const& worldSize,
    Desc&& existentData,
    bool& overlappingCheckSuccessful) const
{
    overlappingCheckSuccessful = true;
    SpaceCalculator spaceCalculator(worldSize);
    std::unordered_map<IntVector2D, std::vector<RealVector2D>> cellPosBySlot;

    // Create map for overlapping check
    if (parameters._overlappingCheck) {
        for (auto const& object : input._objects) {
            auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(object._pos));
            cellPosBySlot[intPos].emplace_back(object._pos);
        }
    }

    // Do multiplication
    Desc result = input;
    auto& numberGen = NumberGenerator::get();
    for (int i = 0; i < parameters._number; ++i) {
        bool overlapping = false;
        Desc copy;
        int attempts = 0;
        do {
            copy = input;
            shift(copy, {toFloat(numberGen.getRandomDouble(0, toInt(worldSize.x))), toFloat(numberGen.getRandomDouble(0, toInt(worldSize.y)))});
            rotate(copy, toInt(numberGen.getRandomDouble(parameters._minAngle, parameters._maxAngle)));
            accelerate(
                copy,
                {toFloat(numberGen.getRandomDouble(parameters._minVelX, parameters._maxVelX)),
                 toFloat(numberGen.getRandomDouble(parameters._minVelY, parameters._maxVelY))},
                toFloat(numberGen.getRandomDouble(parameters._minAngularVel, parameters._maxAngularVel)));

            //overlapping check
            overlapping = false;
            if (parameters._overlappingCheck) {
                for (auto const& object : copy._objects) {
                    auto pos = spaceCalculator.getCorrectedPosition(object._pos);
                    if (isCellPresent(cellPosBySlot, spaceCalculator, pos, 2.0f)) {
                        overlapping = true;
                    }
                }
            }
            ++attempts;
        } while (overlapping && attempts < 200 && overlappingCheckSuccessful);
        if (attempts == 200) {
            overlappingCheckSuccessful = false;
        }

        // Add copy to existentData for overlapping check
        if (parameters._overlappingCheck) {
            for (auto const& object : copy._objects) {
                existentData._objects.emplace_back(object);
                auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(object._pos));
                cellPosBySlot[intPos].emplace_back(object._pos);
            }
        }

        result.add(std::move(copy));
    }

    return result;
}

void DescEditService::addIfSpaceAvailable(Desc& result, Occupancy& cellOccupancy, Desc const& toAdd, float distance, IntVector2D const& worldSize) const
{
    SpaceCalculator space(worldSize);

    for (auto const& object : toAdd._objects) {
        if (!isCellPresent(cellOccupancy, space, object._pos, distance)) {
            result._objects.emplace_back(object);
            cellOccupancy[toIntVector2D(object._pos)].emplace_back(object._pos);
        }
    }
}

void DescEditService::flattenTopology(Desc& description, IntVector2D const& worldSize) const
{
    SpaceCalculator space(worldSize);
    auto cache = description.createCache();

    std::unordered_set<uint64_t> finishedCellIds;
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
        finishedCellIds.insert(workingCellIds.begin(), workingCellIds.end());
        workingCellIds = newWorkingCellIds;
    }
}

void DescEditService::reconnectCells(Desc& description, float maxDistance) const
{
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> cellIndicesBySlot;

    int index = 0;
    for (auto& object : description._objects) {
        object._connections.clear();
        cellIndicesBySlot[toInt(object._pos.x)][toInt(object._pos.y)].emplace_back(toInt(index));
        ++index;
    }

    std::unordered_map<uint64_t, int> objectIdToIndex;
    auto cache = description.createCache();
    for (auto& object : description._objects) {
        auto nearbyCellIndices = getObjectIndicesWithinRadius(description, cellIndicesBySlot, object._pos, maxDistance);
        for (auto const& nearbyCellIndex : nearbyCellIndices) {
            auto const& nearbyCell = description._objects.at(nearbyCellIndex);
            if (object._id != nearbyCell._id && object._connections.size() < MAX_OBJECT_CONNECTIONS && nearbyCell._connections.size() < MAX_OBJECT_CONNECTIONS
                && !object.isConnectedTo(nearbyCell._id)) {
                description.addConnection(object._id, nearbyCell._id, cache);
            }
        }
    }
}

void DescEditService::randomizeCellColors(Desc& description, std::vector<int> const& colorCodes) const
{
    // Step 1: Create random color value for each creature
    std::unordered_map<uint64_t, float> cellColorsByCreatureId;
    for (auto const& creature : description._creatures) {
        cellColorsByCreatureId[creature._id] = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
    }

    // Step 2: Iterate over cells and apply stored color values (including cells without creatureId)
    auto nonCreatureColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
    for (auto& object : description._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            auto it = cellColorsByCreatureId.find(object.getCellRef()._creatureId);
            if (it != cellColorsByCreatureId.end()) {
                object._color = it->second;
            }
        } else {
            object._color = nonCreatureColor;
        }
    }
}

void DescEditService::randomizeGenomeColors(Desc& description, std::vector<int> const& colorCodes) const
{
    for (auto& genome : description._genomes) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& gene : genome._genes) {
            for (auto& node : gene._nodes) {
                node._color = newColor;
            }
        }
    }
}

void DescEditService::randomizeEnergies(Desc& description, float minEnergy, float maxEnergy) const
{
    // Step 1: Create random energy value for each creature
    std::unordered_map<uint64_t, float> creatureEnergies;
    for (auto const& creature : description._creatures) {
        creatureEnergies[creature._id] = NumberGenerator::get().getRandomFloat(toFloat(minEnergy), toFloat(maxEnergy));
    }

    // Step 2: Iterate over cells and apply stored energy values (including cells without creatureId)
    auto const nonCreatureEnergy = NumberGenerator::get().getRandomFloat(toFloat(minEnergy), toFloat(maxEnergy));
    for (auto& object : description._objects) {
        auto type = object.getObjectType();
        if (type == ObjectType_Cell) {
            auto it = creatureEnergies.find(object.getCellRef()._creatureId);
            if (it != creatureEnergies.end()) {
                object.getCellRef()._usableEnergy = it->second;
            }
        } else if (type == ObjectType_FreeCell) {
            object.getFreeCellRef()._energy = nonCreatureEnergy;
        } else if (type == ObjectType_Structure) {
            object.getStructureRef()._energy = nonCreatureEnergy;
        }
    }
}

void DescEditService::randomizeAges(Desc& description, int minAge, int maxAge) const
{
    // Step 1: Create random age value for each creature
    std::unordered_map<uint64_t, int> creatureAges;
    for (auto const& creature : description._creatures) {
        creatureAges[creature._id] = static_cast<int>(NumberGenerator::get().getRandomFloat(toFloat(minAge), toFloat(maxAge)));
    }

    // Step 2: Iterate over cells and apply stored age values (including cells without creatureId)
    auto nonCreatureAge = static_cast<int>(NumberGenerator::get().getRandomFloat(toFloat(minAge), toFloat(maxAge)));
    for (auto& object : description._objects) {
        auto type = object.getObjectType();
        if (type == ObjectType_Cell) {
            auto it = creatureAges.find(object.getCellRef()._creatureId);
            if (it != creatureAges.end()) {
                object.getCellRef()._age = it->second;
            }
        } else if (type == ObjectType_FreeCell) {
            object.getFreeCellRef()._age = nonCreatureAge;
        }
    }
}

void DescEditService::randomizeCountdowns(Desc& description, int minValue, int maxValue) const
{
    // Step 1: Create random countdown value for each creature
    std::unordered_map<uint64_t, int> creatureCountdowns;
    for (auto const& creature : description._creatures) {
        creatureCountdowns[creature._id] = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue)));
    }

    // Step 2: Iterate over cells and apply stored countdown values (including cells without creatureId)
    auto nonCreatureCountdown = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue)));
    for (auto& object : description._objects) {
        if (object.getObjectType() != ObjectType_Cell) {
            continue;
        }
        if (object.getCellRef().getCellType() == CellType_Detonator) {
            auto it = creatureCountdowns.find(object.getCellRef()._creatureId);
            if (it != creatureCountdowns.end()) {
                std::get<DetonatorDesc>(object.getCellRef()._cellType)._countdown = it->second;
            }
        }
    }
}

void DescEditService::randomizeLineageIds(Desc& description) const
{
    for (auto& genome : description._genomes) {
        genome._lineageId = NumberGenerator::get().getRandomInt();
    }
}

void DescEditService::randomizeGlow(Desc& description, float minGlow, float maxGlow) const
{
    for (auto& object : description._objects) {
        if (object.getObjectType() == ObjectType_Structure) {
            object.getStructureRef()._glow = NumberGenerator::get().getRandomFloat(minGlow, maxGlow);
        }
    }
}

void DescEditService::setCenter(Desc& description, RealVector2D const& center) const
{
    auto origCenter = calcCenter(description);
    auto delta = center - origCenter;
    shift(description, delta);
}

RealVector2D DescEditService::calcCenter(Desc const& description) const
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

void DescEditService::shift(Desc& description, RealVector2D const& delta) const
{
    for (auto& object : description._objects) {
        object._pos += delta;
    }

    for (auto& energyParticle : description._energies) {
        energyParticle._pos += delta;
    }
}

void DescEditService::rotate(Desc& description, float angle) const
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

void DescEditService::accelerate(Desc& description, RealVector2D const& velDelta, float angularVelDelta) const
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

void DescEditService::removeCell(Desc& description, uint64_t objectId) const
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

void DescEditService::removeCellIf(Desc& description, std::function<bool(ObjectDesc const&)> const& predicate) const
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

std::vector<ExtendedObjectOrEnergyDesc> DescEditService::getObjects(Desc const& description) const
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

std::vector<ExtendedObjectOrEnergyDesc> DescEditService::getCreatureRepresentatives(Desc const& description) const
{
    return {};
}
