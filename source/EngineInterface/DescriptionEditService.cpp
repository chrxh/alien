#include "DescriptionEditService.h"

#include <cmath>
#include <unordered_map>

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include <Base/Math.h>
#include <Base/Physics.h>

#include <EngineInterface/NumberGenerator.h>

#include "GenomeDescription.h"
#include "SpaceCalculator.h"

Description DescriptionEditService::createRect(CreateRectParameters const& parameters) const
{
    Description result;
    for (int i = 0; i < parameters._width; ++i) {
        for (int j = 0; j < parameters._height; ++j) {
            result._objects.emplace_back(ObjectDescription()
                                           .pos({toFloat(i) * parameters._cellDistance, toFloat(j) * parameters._cellDistance})
                                           .stiffness(parameters._stiffness)
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky)
                                           .type(CellDescription().usableEnergy(parameters._usableEnergy).rawEnergy(parameters._rawEnergy).cellType(parameters._cellType)));
        }
    }
    reconnectCells(result, parameters._cellDistance * 1.1f);
    setCenter(result, parameters._center);
    return result;
}

Description DescriptionEditService::createHex(CreateHexParameters const& parameters) const
{
    Description result;
    auto incY = sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < parameters._layers; ++j) {
        for (int i = -(parameters._layers - 1); i < parameters._layers - j; ++i) {

            //create cell: upper layer
            result._objects.emplace_back(ObjectDescription()
                                           .stiffness(parameters._stiffness)
                                           .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(-j * incY)})
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky)
                                           .type(CellDescription().usableEnergy(parameters._usableEnergy).cellType(parameters._cellType)));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                result._objects.emplace_back(ObjectDescription()
                                               .stiffness(parameters._stiffness)
                                               .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(j * incY)})
                                               .color(parameters._color)
                                               .fixed(parameters._fixed)
                                               .type(CellDescription().usableEnergy(parameters._usableEnergy).cellType(StructureDescription())));
            }
        }
    }

    reconnectCells(result, parameters._cellDistance * 1.5f);
    setCenter(result, parameters._center);

    return result;
}

Description DescriptionEditService::createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters) const
{
    Description result;

    if (parameters._radius <= 1 + NEAR_ZERO) {
        result._objects.emplace_back(ObjectDescription()
                                       .pos(parameters._center)
                                       .stiffness(parameters._stiffness)
                                       .color(parameters._color)
                                       .fixed(parameters._fixed)
                                       .sticky(parameters._sticky)
                                       .type(CellDescription().usableEnergy(parameters._usableEnergy).cellType(StructureDescription())));
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
            result._objects.emplace_back(ObjectDescription()
                                           .stiffness(parameters._stiffness)
                                           .pos({parameters._center.x + dxMod, parameters._center.y + dy})
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky)
                                           .type(CellDescription().usableEnergy(parameters._usableEnergy).cellType(StructureDescription())));
        }
    }
    return result;
}

namespace
{
    void topologyCorrection(SpaceCalculator const& spaceCalc, Description& description, uint64_t creatureId)
    {
        ObjectDescription* refCell = nullptr;
        for (auto& object : description._objects) {
            if (object.getCellRef()._creatureId == creatureId) {
                if (!refCell) {
                    refCell = &object;
                }
                auto topologyCorrection = spaceCalc.getCorrectionIncrement(refCell->_pos, object._pos);
                object._pos = object._pos + topologyCorrection;
            }
        }
    }

    void correctConnectionsForNonCreatures(Description& description, IntVector2D const& worldSize)
    {
        auto threshold = std::min(worldSize.x, worldSize.y) / 3;
        auto cache = description.createCache();

        for (auto& object : description._objects) {
            if (object.getCellRef()._creatureId.has_value()) {
                continue;
            }
            std::vector<ConnectionDescription> newConnections;
            float angleToAdd = 0;
            for (auto connection : object._connections) {
                auto const& connectingCell = description.getObjectRef(connection._objectId, cache);
                if (Math::length(object._pos - connectingCell._pos) > threshold) {
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

void DescriptionEditService::duplicate(Description& description, IntVector2D const& origSize, IntVector2D const& size) const
{
    correctConnectionsForNonCreatures(description, origSize);

    SpaceCalculator spaceCalc(origSize);

    Description result;
    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            auto clone = Description(description);
            clone.assignNewIds();
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

    description = Description(result);
}

namespace
{
    std::vector<int> getObjectIndicesWithinRadius(
        Description const& description,
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

Description DescriptionEditService::gridMultiply(Description const& input, GridMultiplyParameters const& parameters) const
{
    Description result;
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

Description DescriptionEditService::randomMultiply(
    Description const& input,
    RandomMultiplyParameters const& parameters,
    IntVector2D const& worldSize,
    Description&& existentData,
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
    Description result = input;
    auto& numberGen = NumberGenerator::get();
    for (int i = 0; i < parameters._number; ++i) {
        bool overlapping = false;
        Description copy;
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

void DescriptionEditService::addIfSpaceAvailable(
    Description& result,
    Occupancy& cellOccupancy,
    Description const& toAdd,
    float distance,
    IntVector2D const& worldSize) const
{
    SpaceCalculator space(worldSize);

    for (auto const& object : toAdd._objects) {
        if (!isCellPresent(cellOccupancy, space, object._pos, distance)) {
            result._objects.emplace_back(object);
            cellOccupancy[toIntVector2D(object._pos)].emplace_back(object._pos);
        }
    }
}

void DescriptionEditService::flattenTopology(Description& description, IntVector2D const& worldSize) const
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
                    auto& otherCell = description.getObjectRef(connection._objectId, cache);
                    otherCell._pos += space.getCorrectionIncrement(object._pos, otherCell._pos);

                    freeCellIds.erase(connection._objectId);
                    newWorkingCellIds.insert(connection._objectId);
                }
            }
        }
        finishedCellIds.insert(workingCellIds.begin(), workingCellIds.end());
        workingCellIds = newWorkingCellIds;
    }
}

void DescriptionEditService::reconnectCells(Description& description, float maxDistance) const
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
            if (object._id != nearbyCell._id && object._connections.size() < MAX_CELL_BONDS && nearbyCell._connections.size() < MAX_CELL_BONDS
                && !object.isConnectedTo(nearbyCell._id)) {
                description.addConnection(object._id, nearbyCell._id, cache);
            }
        }
    }
}

void DescriptionEditService::randomizeCellColors(Description& description, std::vector<int> const& colorCodes) const
{
    // Step 1: Create random color value for each creature
    std::unordered_map<uint64_t, float> cellColorsByCreatureId;
    for (auto const& creature : description._creatures) {
        cellColorsByCreatureId[creature._id] = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
    }

    // Step 2: Iterate over cells and apply stored color values (including cells without creatureId)
    auto nonCreatureColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
    for (auto& object : description._objects) {
        if (object.getCellRef()._creatureId.has_value()) {
            auto it = cellColorsByCreatureId.find(object.getCellRef()._creatureId.value());
            if (it != cellColorsByCreatureId.end()) {
                object._color = it->second;
            }
        } else {
            object.getCellRef()._usableEnergy = nonCreatureColor;
        }
    }
}

void DescriptionEditService::randomizeGenomeColors(Description& description, std::vector<int> const& colorCodes) const
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

void DescriptionEditService::randomizeEnergies(Description& description, float minEnergy, float maxEnergy) const
{
    // Step 1: Create random energy value for each creature
    std::unordered_map<uint64_t, float> creatureEnergies;
    for (auto const& creature : description._creatures) {
        creatureEnergies[creature._id] = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
    }
    
    // Step 2: Iterate over cells and apply stored energy values (including cells without creatureId)
    auto nonCreatureEnergy = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
    for (auto& object : description._objects) {
        if (object.getCellRef()._creatureId.has_value()) {
            auto it = creatureEnergies.find(object.getCellRef()._creatureId.value());
            if (it != creatureEnergies.end()) {
                object.getCellRef()._usableEnergy = it->second;
            }
        } else {
            object.getCellRef()._usableEnergy = nonCreatureEnergy;
        }
    }
}

void DescriptionEditService::randomizeAges(Description& description, int minAge, int maxAge) const
{
    // Step 1: Create random age value for each creature
    std::unordered_map<uint64_t, int> creatureAges;
    for (auto const& creature : description._creatures) {
        creatureAges[creature._id] = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minAge), toDouble(maxAge)));
    }
    
    // Step 2: Iterate over cells and apply stored age values (including cells without creatureId)
    auto nonCreatureAge = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minAge), toDouble(maxAge)));
    for (auto& object : description._objects) {
        if (object.getCellRef()._creatureId.has_value()) {
            auto it = creatureAges.find(object.getCellRef()._creatureId.value());
            if (it != creatureAges.end()) {
                object.getCellRef()._age = it->second;
            }
        } else {
            object.getCellRef()._age = nonCreatureAge;
        }
    }
}

void DescriptionEditService::randomizeCountdowns(Description& description, int minValue, int maxValue) const
{
    // Step 1: Create random countdown value for each creature
    std::unordered_map<uint64_t, int> creatureCountdowns;
    for (auto const& creature : description._creatures) {
        creatureCountdowns[creature._id] = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue)));
    }
    
    // Step 2: Iterate over cells and apply stored countdown values (including cells without creatureId)
    auto nonCreatureCountdown = static_cast<int>(NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue)));
    for (auto& object : description._objects) {
        if (object.getCellRef().getCellType() == CellType_Detonator) {
            if (object.getCellRef()._creatureId.has_value()) {
                auto it = creatureCountdowns.find(object.getCellRef()._creatureId.value());
                if (it != creatureCountdowns.end()) {
                    std::get<DetonatorDescription>(object.getCellRef()._cellType)._countdown = it->second;
                }
            } else {
                std::get<DetonatorDescription>(object.getCellRef()._cellType)._countdown = nonCreatureCountdown;
            }
        }
    }
}

void DescriptionEditService::randomizeLineageIds(Description& description) const
{
    for (auto& creature : description._creatures) {
        creature._lineageId = NumberGenerator::get().getRandomInt();
    }
}

void DescriptionEditService::setCenter(Description& description, RealVector2D const& center) const
{
    auto origCenter = calcCenter(description);
    auto delta = center - origCenter;
    shift(description, delta);
}

RealVector2D DescriptionEditService::calcCenter(Description const& description) const
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

void DescriptionEditService::shift(Description& description, RealVector2D const& delta) const
{
    for (auto& object : description._objects) {
        object._pos += delta;
    }

    for (auto& energyParticle : description._energies) {
        energyParticle._pos += delta;
    }
}

void DescriptionEditService::rotate(Description& description, float angle) const
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

void DescriptionEditService::accelerate(Description& description, RealVector2D const& velDelta, float angularVelDelta) const
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

void DescriptionEditService::removeCell(Description& description, uint64_t objectId) const
{
    std::erase_if(description._objects, [&](auto const& object) { return object._id == objectId; });

    // Check if any creatures have no cells left
    std::unordered_set<uint64_t> creaturesWithCells;
    for (auto const& object : description._objects) {
        if (object.getCellRef()._creatureId.has_value()) {
            creaturesWithCells.insert(object.getCellRef()._creatureId.value());
        }
    }
    std::erase_if(description._creatures, [&](auto const& creature) { return !creaturesWithCells.contains(creature._id); });

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

void DescriptionEditService::removeCellIf(Description& description, std::function<bool(ObjectDescription const&)> const& predicate) const
{
    std::unordered_set<uint64_t> removedCellIds;
    auto extPredicate = [&](ObjectDescription const& object) {
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
        if (object.getCellRef()._creatureId.has_value()) {
            creaturesWithCells.insert(object.getCellRef()._creatureId.value());
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

bool DescriptionEditService::isCellPresent(
    Occupancy const& cellPosBySlot,
    SpaceCalculator const& spaceCalculator,
    RealVector2D const& posToCheck,
    float distance) const
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

uint64_t DescriptionEditService::getId(ExtendedObjectOrEnergyDescription const& entity) const
{
    if (std::holds_alternative<ExtendedObjectDescription>(entity)) {
        return std::get<ExtendedObjectDescription>(entity).object._id;
    }
    return std::get<EnergyDescription>(entity)._id;
}

RealVector2D DescriptionEditService::getPos(ExtendedObjectOrEnergyDescription const& entity) const
{
    if (std::holds_alternative<ExtendedObjectDescription>(entity)) {
        return std::get<ExtendedObjectDescription>(entity).object._pos;
    }
    return std::get<EnergyDescription>(entity)._pos;
}

std::vector<ExtendedObjectOrEnergyDescription> DescriptionEditService::getObjects(Description const& description) const
{
    std::vector<ExtendedObjectOrEnergyDescription> result;
    for (auto const& energyParticle : description._energies) {
        result.emplace_back(energyParticle);
    }

    // Build a map of creatureId to genome
    std::unordered_map<uint64_t, GenomeDescription> genomeByCreatureId;
    for (auto const& creature : description._creatures) {
        auto genomeIt =
            std::find_if(description._genomes.begin(), description._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
        if (genomeIt != description._genomes.end()) {
            genomeByCreatureId.emplace(creature._id, *genomeIt);
        }
    }

    for (auto const& object : description._objects) {
        ExtendedObjectDescription extCell;
        extCell.object = object;
        extCell.creatureId = object.getCellRef()._creatureId;
        if (object.getCellRef()._creatureId.has_value()) {
            auto genomeIt = genomeByCreatureId.find(object.getCellRef()._creatureId.value());
            if (genomeIt != genomeByCreatureId.end()) {
                extCell.genome = genomeIt->second;
            }
        }
        result.emplace_back(extCell);
    }
    return result;
}

std::vector<ExtendedObjectOrEnergyDescription> DescriptionEditService::getCellsForCreatureRepresentatives(Description const& description) const
{
    return {};
}
