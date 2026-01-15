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
            result._cells.emplace_back(CellDescription()
                                           .pos({toFloat(i) * parameters._cellDistance, toFloat(j) * parameters._cellDistance})
                                           .usableEnergy(parameters._usableEnergy)
                                           .rawEnergy(parameters._rawEnergy)
                                           .stiffness(parameters._stiffness)
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky)
                                           .cellType(parameters._cellType));
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
            result._cells.emplace_back(CellDescription()
                                           .cellType(StructureCellDescription())
                                           .usableEnergy(parameters._usableEnergy)
                                           .stiffness(parameters._stiffness)
                                           .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(-j * incY)})
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky)
                                           .cellType(parameters._cellType));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                result._cells.emplace_back(CellDescription()
                                               .cellType(StructureCellDescription())
                                               .usableEnergy(parameters._usableEnergy)
                                               .stiffness(parameters._stiffness)
                                               .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(j * incY)})
                                               .color(parameters._color)
                                               .fixed(parameters._fixed));
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
        result._cells.emplace_back(CellDescription()
                                       .cellType(StructureCellDescription())
                                       .pos(parameters._center)
                                       .usableEnergy(parameters._usableEnergy)
                                       .stiffness(parameters._stiffness)
                                       .color(parameters._color)
                                       .fixed(parameters._fixed)
                                       .sticky(parameters._sticky));
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
            result._cells.emplace_back(CellDescription()
                                           .cellType(StructureCellDescription())
                                           .usableEnergy(parameters._usableEnergy)
                                           .stiffness(parameters._stiffness)
                                           .pos({parameters._center.x + dxMod, parameters._center.y + dy})
                                           .color(parameters._color)
                                           .fixed(parameters._fixed)
                                           .sticky(parameters._sticky));
        }
    }
    return result;
}

namespace
{
    void topologyCorrection(SpaceCalculator const& spaceCalc, Description& description, uint64_t creatureId)
    {
        CellDescription* refCell = nullptr;
        for (auto& cell : description._cells) {
            if (cell._creatureId == creatureId) {
                if (!refCell) {
                    refCell = &cell;
                }
                auto topologyCorrection = spaceCalc.getCorrectionIncrement(refCell->_pos, cell._pos);
                cell._pos = cell._pos + topologyCorrection;
            }
        }
    }

    void correctConnectionsForNonCreatures(Description& description, IntVector2D const& worldSize)
    {
        auto threshold = std::min(worldSize.x, worldSize.y) / 3;
        auto cache = description.createCache();

        for (auto& cell : description._cells) {
            if (cell._creatureId.has_value()) {
                continue;
            }
            std::vector<ConnectionDescription> newConnections;
            float angleToAdd = 0;
            for (auto connection : cell._connections) {
                auto const& connectingCell = description.getCellRef(connection._cellId, cache);
                if (Math::length(cell._pos - connectingCell._pos) > threshold) {
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
            cell._connections = newConnections;
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
            for (auto& cell : clone._cells) {
                cell._pos = RealVector2D{cell._pos.x + incX, cell._pos.y + incY};
                result._cells.emplace_back(cell);
            }
            for (auto particle : clone._particles) {
                auto origPos = particle._pos;
                particle._pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                result._particles.emplace_back(particle);
            }
            result._genomes.insert(result._genomes.end(), clone._genomes.begin(), clone._genomes.end());
        }
    }

    description = Description(result);
}

namespace
{
    std::vector<int> getCellIndicesWithinRadius(
        Description const& description,
        std::unordered_map<int, std::unordered_map<int, std::vector<int>>> const& cellIndicesBySlot,
        RealVector2D const& pos,
        float radius)
    {
        std::vector<int> result;
        IntVector2D upperLeftIntPos{toInt(pos.x - radius - 0.5f), toInt(pos.y - radius - 0.5f)};
        IntVector2D lowerRightIntPos{toInt(pos.x + radius + 0.5f), toInt(pos.y + radius + 0.5f)};
        for (int x = upperLeftIntPos.x; x <= lowerRightIntPos.x; ++x) {
            for (int y = upperLeftIntPos.y; y <= lowerRightIntPos.y; ++y) {
                if (cellIndicesBySlot.find(x) != cellIndicesBySlot.end()) {
                    if (cellIndicesBySlot.at(x).find(y) != cellIndicesBySlot.at(x).end()) {
                        for (auto const& cellIndex : cellIndicesBySlot.at(x).at(y)) {
                            auto const& cell = description._cells.at(cellIndex);
                            if (Math::length(cell._pos - pos) <= radius) {
                                result.emplace_back(cellIndex);
                            }
                        }
                    }
                }
            }
        }
        std::sort(result.begin(), result.end(), [&](int index1, int index2) {
            auto const& cell1 = description._cells.at(index1);
            auto const& cell2 = description._cells.at(index2);
            return Math::length(cell1._pos - pos) < Math::length(cell2._pos - pos);
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
        for (auto const& cell : input._cells) {
            auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
            cellPosBySlot[intPos].emplace_back(cell._pos);
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
                for (auto const& cell : copy._cells) {
                    auto pos = spaceCalculator.getCorrectedPosition(cell._pos);
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
            for (auto const& cell : copy._cells) {
                existentData._cells.emplace_back(cell);
                auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
                cellPosBySlot[intPos].emplace_back(cell._pos);
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

    for (auto const& cell : toAdd._cells) {
        if (!isCellPresent(cellOccupancy, space, cell._pos, distance)) {
            result._cells.emplace_back(cell);
            cellOccupancy[toIntVector2D(cell._pos)].emplace_back(cell._pos);
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

    for (auto const& cell : description._cells) {
        freeCellIds.insert(cell._id);
    }
    while (!workingCellIds.empty() || !freeCellIds.empty()) {

        // Take an arbitrary cell to start with
        if (workingCellIds.empty()) {
            workingCellIds.insert(*freeCellIds.begin());
            freeCellIds.erase(freeCellIds.begin());
        }

        // Process working cells: find connected free cells and correct topology
        std::unordered_set<uint64_t> newWorkingCellIds;
        for (auto const& cellId : workingCellIds) {
            auto& cell = description.getCellRef(cellId, cache);

            for (auto const& connection : cell._connections) {
                if (freeCellIds.contains(connection._cellId)) {
                    // Do topology correction
                    auto& otherCell = description.getCellRef(connection._cellId, cache);
                    otherCell._pos += space.getCorrectionIncrement(cell._pos, otherCell._pos);

                    freeCellIds.erase(connection._cellId);
                    newWorkingCellIds.insert(connection._cellId);
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
    for (auto& cell : description._cells) {
        cell._connections.clear();
        cellIndicesBySlot[toInt(cell._pos.x)][toInt(cell._pos.y)].emplace_back(toInt(index));
        ++index;
    }

    std::unordered_map<uint64_t, int> cellIdToIndex;
    auto cache = description.createCache();
    for (auto& cell : description._cells) {
        auto nearbyCellIndices = getCellIndicesWithinRadius(description, cellIndicesBySlot, cell._pos, maxDistance);
        for (auto const& nearbyCellIndex : nearbyCellIndices) {
            auto const& nearbyCell = description._cells.at(nearbyCellIndex);
            if (cell._id != nearbyCell._id && cell._connections.size() < MAX_CELL_BONDS && nearbyCell._connections.size() < MAX_CELL_BONDS
                && !cell.isConnectedTo(nearbyCell._id)) {
                description.addConnection(cell._id, nearbyCell._id, cache);
            }
        }
    }
}

void DescriptionEditService::randomizeCellColors(Description& description, std::vector<int> const& colorCodes) const
{
    for (auto const& creature : description._creatures) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : description._cells) {
            if (cell._creatureId == creature._id) {
                cell._color = newColor;
            }
        }
    }
    {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : description._cells) {
            if (!cell._creatureId.has_value()) {
                cell._color = newColor;
            }
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
    for (auto& cell : description._cells) {
        if (cell._creatureId.has_value()) {
            auto it = creatureEnergies.find(cell._creatureId.value());
            if (it != creatureEnergies.end()) {
                cell._usableEnergy = it->second;
            } else {
                // Handle cells with creatureId not in the map (shouldn't normally happen, but handle gracefully)
                cell._usableEnergy = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
            }
        } else {
            cell._usableEnergy = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
        }
    }
}

void DescriptionEditService::randomizeAges(Description& description, int minAge, int maxAge) const
{
    // Step 1: Create random age value for each creature
    std::unordered_map<uint64_t, int> creatureAges;
    for (auto const& creature : description._creatures) {
        creatureAges[creature._id] = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minAge), static_cast<uint32_t>(maxAge)));
    }
    
    // Step 2: Iterate over cells and apply stored age values (including cells without creatureId)
    for (auto& cell : description._cells) {
        if (cell._creatureId.has_value()) {
            auto it = creatureAges.find(cell._creatureId.value());
            if (it != creatureAges.end()) {
                cell._age = it->second;
            } else {
                // Handle cells with creatureId not in the map (shouldn't normally happen, but handle gracefully)
                cell._age = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minAge), static_cast<uint32_t>(maxAge)));
            }
        } else {
            cell._age = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minAge), static_cast<uint32_t>(maxAge)));
        }
    }
}

void DescriptionEditService::randomizeCountdowns(Description& description, int minValue, int maxValue) const
{
    // Step 1: Create random countdown value for each creature
    std::unordered_map<uint64_t, int> creatureCountdowns;
    for (auto const& creature : description._creatures) {
        creatureCountdowns[creature._id] = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minValue), static_cast<uint32_t>(maxValue)));
    }
    
    // Step 2: Iterate over cells and apply stored countdown values (including cells without creatureId)
    for (auto& cell : description._cells) {
        if (cell.getCellType() == CellType_Detonator) {
            if (cell._creatureId.has_value()) {
                auto it = creatureCountdowns.find(cell._creatureId.value());
                if (it != creatureCountdowns.end()) {
                    std::get<DetonatorDescription>(cell._cellType)._countdown = it->second;
                } else {
                    // Handle cells with creatureId not in the map (shouldn't normally happen, but handle gracefully)
                    std::get<DetonatorDescription>(cell._cellType)._countdown = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minValue), static_cast<uint32_t>(maxValue)));
                }
            } else {
                std::get<DetonatorDescription>(cell._cellType)._countdown = static_cast<int>(NumberGenerator::get().getRandomInt(static_cast<uint32_t>(minValue), static_cast<uint32_t>(maxValue)));
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
    auto numEntities = description._cells.size() + description._particles.size();
    for (auto const& cell : description._cells) {
        result += cell._pos;
    }

    for (auto const& particle : description._particles) {
        result += particle._pos;
    }
    result /= numEntities;
    return result;
}

void DescriptionEditService::shift(Description& description, RealVector2D const& delta) const
{
    for (auto& cell : description._cells) {
        cell._pos += delta;
    }

    for (auto& particle : description._particles) {
        particle._pos += delta;
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
    for (auto& cell : description._cells) {
        rotate(cell._pos);
    }
    for (auto& particle : description._particles) {
        rotate(particle._pos);
    }
}

void DescriptionEditService::accelerate(Description& description, RealVector2D const& velDelta, float angularVelDelta) const
{
    auto center = calcCenter(description);

    auto accelerate = [&](RealVector2D const& pos, RealVector2D& vel) {
        auto relPos = pos - center;
        vel += Physics::tangentialVelocity(relPos, velDelta, angularVelDelta);
    };
    for (auto& cell : description._cells) {
        accelerate(cell._pos, cell._vel);
    }
    for (auto& particle : description._particles) {
        accelerate(particle._pos, particle._vel);
    }
}

void DescriptionEditService::removeCell(Description& description, uint64_t cellId) const
{
    std::erase_if(description._cells, [&](auto const& cell) { return cell._id == cellId; });

    // Check if any creatures have no cells left
    std::unordered_set<uint64_t> creaturesWithCells;
    for (auto const& cell : description._cells) {
        if (cell._creatureId.has_value()) {
            creaturesWithCells.insert(cell._creatureId.value());
        }
    }
    std::erase_if(description._creatures, [&](auto const& creature) { return !creaturesWithCells.contains(creature._id); });

    for (auto& cell : description._cells) {
        for (int i = 0, numConnections = cell._connections.size(); i < numConnections; ++i) {
            auto const& connection = cell._connections[i];
            if (connection._cellId == cellId) {
                auto angleToAdd = connection._angleFromPrevious;
                for (int k = i; k < numConnections - 1; ++k) {
                    cell._connections.at(k) = cell._connections.at(k + 1);
                }

                if (i < numConnections - 1) {
                    cell._connections.at(i)._angleFromPrevious += angleToAdd;
                } else {
                    cell._connections.at(0)._angleFromPrevious += angleToAdd;
                }

                cell._connections.pop_back();
                return;
            }
        }
    }
}

void DescriptionEditService::removeCellIf(Description& description, std::function<bool(CellDescription const&)> const& predicate) const
{
    std::unordered_set<uint64_t> removedCellIds;
    auto extPredicate = [&](CellDescription const& cell) {
        auto result = predicate(cell);
        if (result) {
            removedCellIds.insert(cell._id);
        }
        return result;
    };

    std::erase_if(description._cells, extPredicate);

    // Check if any creatures have no cells left
    std::unordered_set<uint64_t> creaturesWithCells;
    for (auto const& cell : description._cells) {
        if (cell._creatureId.has_value()) {
            creaturesWithCells.insert(cell._creatureId.value());
        }
    }
    std::erase_if(description._creatures, [&](auto const& creature) { return !creaturesWithCells.contains(creature._id); });

    for (auto& cell : description._cells) {
        for (int i = 0, numConnections = cell._connections.size(); i < numConnections; ++i) {
            auto const& connection = cell._connections[i];
            if (removedCellIds.contains(connection._cellId)) {
                auto angleToAdd = connection._angleFromPrevious;
                for (int k = i; k < numConnections - 1; ++k) {
                    cell._connections.at(k) = cell._connections.at(k + 1);
                }

                if (i < numConnections - 1) {
                    cell._connections.at(i)._angleFromPrevious += angleToAdd;
                } else {
                    cell._connections.at(0)._angleFromPrevious += angleToAdd;
                }

                cell._connections.pop_back();
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

uint64_t DescriptionEditService::getId(ExtendedCellOrParticleDescription const& entity) const
{
    if (std::holds_alternative<ExtendedCellDescription>(entity)) {
        return std::get<ExtendedCellDescription>(entity).cell._id;
    }
    return std::get<ParticleDescription>(entity)._id;
}

RealVector2D DescriptionEditService::getPos(ExtendedCellOrParticleDescription const& entity) const
{
    if (std::holds_alternative<ExtendedCellDescription>(entity)) {
        return std::get<ExtendedCellDescription>(entity).cell._pos;
    }
    return std::get<ParticleDescription>(entity)._pos;
}

std::vector<ExtendedCellOrParticleDescription> DescriptionEditService::getObjects(Description const& description) const
{
    std::vector<ExtendedCellOrParticleDescription> result;
    for (auto const& particle : description._particles) {
        result.emplace_back(particle);
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

    for (auto const& cell : description._cells) {
        ExtendedCellDescription extCell;
        extCell.cell = cell;
        extCell.creatureId = cell._creatureId;
        if (cell._creatureId.has_value()) {
            auto genomeIt = genomeByCreatureId.find(cell._creatureId.value());
            if (genomeIt != genomeByCreatureId.end()) {
                extCell.genome = genomeIt->second;
            }
        }
        result.emplace_back(extCell);
    }
    return result;
}

std::vector<ExtendedCellOrParticleDescription> DescriptionEditService::getCellsForCreatureRepresentatives(Description const& description) const
{
    return {};
}
