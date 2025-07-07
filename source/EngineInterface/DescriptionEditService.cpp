#include "DescriptionEditService.h"

#include <cmath>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include "Base/Math.h"
#include "Base/Physics.h"
#include "EngineInterface/NumberGenerator.h"

#include "GenomeDescription.h"
#include "SpaceCalculator.h"

CollectionDescription DescriptionEditService::createRect(CreateRectParameters const& parameters)
{
    CollectionDescription result;
    for (int i = 0; i < parameters._width; ++i) {
        for (int j = 0; j < parameters._height; ++j) {
            result.addCell(CellDescription()
                               .pos({toFloat(i) * parameters._cellDistance, toFloat(j) * parameters._cellDistance})
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky)
                               .cellTypeData(parameters._cellType));
        }
    }
    reconnectCells(result, parameters._cellDistance * 1.1f);
    setCenter(result, parameters._center);
    return result;
}

CollectionDescription DescriptionEditService::createHex(CreateHexParameters const& parameters)
{
    CollectionDescription result;
    auto incY = sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < parameters._layers; ++j) {
        for (int i = -(parameters._layers - 1); i < parameters._layers - j; ++i) {

            //create cell: upper layer
            result.addCell(CellDescription()
                               .cellTypeData(StructureCellDescription())
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(-j * incY)})
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky)
                               .cellTypeData(parameters._cellType));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                result.addCell(CellDescription()
                                   .cellTypeData(StructureCellDescription())
                                   .energy(parameters._energy)
                                   .stiffness(parameters._stiffness)
                                   .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(j * incY)})
                                   .color(parameters._color)
                                   .barrier(parameters._barrier));
            }
        }
    }

    reconnectCells(result, parameters._cellDistance * 1.5f);
    setCenter(result, parameters._center);

    return result;
}

CollectionDescription DescriptionEditService::createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters)
{
    CollectionDescription result;

    if (parameters._radius <= 1 + NEAR_ZERO) {
        result.addCell(CellDescription()
                           .cellTypeData(StructureCellDescription())
                           .pos(parameters._center)
                           .energy(parameters._energy)
                           .stiffness(parameters._stiffness)
                           .color(parameters._color)
                           .barrier(parameters._barrier)
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
            result.addCell(CellDescription()
                               .cellTypeData(StructureCellDescription())
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .pos({parameters._center.x + dxMod, parameters._center.y + dy})
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky));

        }
    }
    return result;
}

namespace
{
    void topologyCorrection(SpaceCalculator const& spaceCalc, CreatureDescription& creature)
    {
        CHECK(!creature._cells.empty());
        auto refCell = creature._cells.front();
        for (auto & cell : creature._cells) {
            auto topologyCorrection = spaceCalc.getCorrectionIncrement(refCell._pos, cell._pos);
            cell._pos = cell._pos + topologyCorrection;
        }
    }

    void correctConnectionsForNonCreatures(CollectionDescription& data, IntVector2D const& worldSize)
    {
        auto threshold = std::min(worldSize.x, worldSize.y) / 3;
        auto cache = data.createCache();

        for (auto& cell : data._cells) {
            std::vector<ConnectionDescription> newConnections;
            float angleToAdd = 0;
            for (auto connection : cell._connections) {
                auto const& connectingCell = data.getCellRef(connection._cellId);
                if (/*spaceCalculator.distance*/ Math::length(cell._pos - connectingCell._pos) > threshold) {
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

void DescriptionEditService::duplicate(CollectionDescription& data, IntVector2D const& origSize, IntVector2D const& size)
{
    correctConnectionsForNonCreatures(data, origSize);

    auto clone = CollectionDescription(data);

    SpaceCalculator spaceCalc(origSize);

    CollectionDescription result;
    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            for (auto creature : clone._creatures) {
                topologyCorrection(spaceCalc, creature);
                auto origPos = calcCenter(creature);
                RealVector2D newPos = {origPos.x + incX, origPos.y + incY};
                if (newPos.x < size.x && newPos.y < size.y) {
                    for (auto& cell : creature._cells) {
                        cell._pos = RealVector2D{cell._pos.x + incX, cell._pos.y + incY};
                        if (incX > 0 || incY > 0) {
                            removeMetadata(cell);
                        }
                    }
                    result._creatures.emplace_back(creature);
                }
            }
            for (auto cell : clone._cells) {
                RealVector2D newPos = {cell._pos.x + incX, cell._pos.y + incY};
                cell._pos = RealVector2D{cell._pos.x + incX, cell._pos.y + incY};
                if (newPos.x < size.x && newPos.y < size.y) {
                    if (incX > 0 || incY > 0) {
                        removeMetadata(cell);
                    }
                    result._cells.emplace_back(cell);
                }
            }
            for (auto particle : data._particles) {
                auto origPos = particle._pos;
                particle._pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                if (particle._pos.x < size.x && particle._pos.y < size.y) {
                    result.addParticle(particle);
                }
            }
        }
    }

    data = CollectionDescription(result);
    assignNewObjectAndCreatureIds(data);
}

namespace
{
    std::vector<int> getCellIndicesWithinRadius(
        CollectionDescription const& data,
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
                            auto const& cell = data._cells.at(cellIndex);
                            if (Math::length(cell._pos - pos) <= radius) {
                                result.emplace_back(cellIndex);
                            }
                        }
                    }
                }
            }
        }
        std::sort(result.begin(), result.end(), [&](int index1, int index2) {
            auto const& cell1 = data._cells.at(index1);
            auto const& cell2 = data._cells.at(index2);
            return Math::length(cell1._pos - pos) < Math::length(cell2._pos - pos);
        });
        return result;
    }
}

CollectionDescription DescriptionEditService::gridMultiply(CollectionDescription const& input, GridMultiplyParameters const& parameters)
{
    CollectionDescription result;
    auto clone = input;
    auto cloneWithoutMetadata = input;
    removeMetadata(cloneWithoutMetadata);
    for (int i = 0; i < parameters._horizontalNumber; ++i) {
        for (int j = 0; j < parameters._verticalNumber; ++j) {
            auto templateData = [&] {
                if (i == 0 && j == 0) {
                    return clone;
                }
                return cloneWithoutMetadata;
            }();
            shift(templateData, {i * parameters._horizontalDistance, j * parameters._verticalDistance});
            rotate(templateData, i * parameters._horizontalAngleInc + j * parameters._verticalAngleInc);
            accelerate(
                templateData,
                {i * parameters._horizontalVelXinc + j * parameters._verticalVelXinc, i * parameters._horizontalVelYinc + j * parameters._verticalVelYinc},
                i * parameters._horizontalAngularVelInc + j * parameters._verticalAngularVelInc);

            result.add(templateData);
        }
    }

    return result;
}

CollectionDescription DescriptionEditService::randomMultiply(
    CollectionDescription const& input,
    RandomMultiplyParameters const& parameters,
    IntVector2D const& worldSize,
    CollectionDescription&& existentData,
    bool& overlappingCheckSuccessful)
{
    overlappingCheckSuccessful = true;
    SpaceCalculator spaceCalculator(worldSize);
    std::unordered_map<IntVector2D, std::vector<RealVector2D>> cellPosBySlot;

    //create map for overlapping check
    if (parameters._overlappingCheck) {
        input.forEach([&](CellDescription const& cell) {
            auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
            cellPosBySlot[intPos].emplace_back(cell._pos);
        });
    }

    //do multiplication
    CollectionDescription result = input;
    auto& numberGen = NumberGenerator::get();
    for (int i = 0; i < parameters._number; ++i) {
        bool overlapping = false;
        CollectionDescription copy;
        int attempts = 0;
        do {
            copy = input;
            removeMetadata(copy);
            shift(copy, {toFloat(numberGen.getRandomDouble(0, toInt(worldSize.x))), toFloat(numberGen.getRandomDouble(0, toInt(worldSize.y)))});
            rotate(copy, toInt(numberGen.getRandomDouble(parameters._minAngle, parameters._maxAngle)));
            accelerate(copy,
                {toFloat(numberGen.getRandomDouble(parameters._minVelX, parameters._maxVelX)),
                 toFloat(numberGen.getRandomDouble(parameters._minVelY, parameters._maxVelY))},
                toFloat(numberGen.getRandomDouble(parameters._minAngularVel, parameters._maxAngularVel)));

            //overlapping check
            overlapping = false;
            if (parameters._overlappingCheck) {
                copy.forEach([&](CellDescription const& cell) {
                    auto pos = spaceCalculator.getCorrectedPosition(cell._pos);
                    if (isCellPresent(cellPosBySlot, spaceCalculator, pos, 2.0f)) {
                        overlapping = true;
                    }
                });
            }
            ++attempts;
        } while (overlapping && attempts < 200 && overlappingCheckSuccessful);
        if (attempts == 200) {
            overlappingCheckSuccessful = false;
        }

        result.add(copy);

        //add copy to existentData for overlapping check
        if (parameters._overlappingCheck) {
            copy.forEach([&](CellDescription const& cell) {
                existentData._cells.emplace_back(cell);
                auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
                cellPosBySlot[intPos].emplace_back(cell._pos);
            });
        }
    }

    return result;
}

void DescriptionEditService::addIfSpaceAvailable(
    CollectionDescription& result,
    Occupancy& cellOccupancy,
    CollectionDescription const& toAdd,
    float distance,
    IntVector2D const& worldSize)
{
    SpaceCalculator space(worldSize);

    for (auto const& cell : toAdd._cells) {
        if (!isCellPresent(cellOccupancy, space, cell._pos, distance)) {
            result.addCell(cell);
            cellOccupancy[toIntVector2D(cell._pos)].emplace_back(cell._pos);
        }
    }
}

void DescriptionEditService::reconnectCells(CollectionDescription& data, float maxDistance)
{
    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> cellIndicesBySlot;

    int index = 0;
    for (auto& cell : data._cells) {
        cell._connections.clear();
        cellIndicesBySlot[toInt(cell._pos.x)][toInt(cell._pos.y)].emplace_back(toInt(index));
        ++index;
    }

    std::unordered_map<uint64_t, int> cellIdToIndex;
    auto cache = data.createCache();
    for (auto& cell : data._cells) {
        auto nearbyCellIndices = getCellIndicesWithinRadius(data, cellIndicesBySlot, cell._pos, maxDistance);
        for (auto const& nearbyCellIndex : nearbyCellIndices) {
            auto const& nearbyCell = data._cells.at(nearbyCellIndex);
            if (cell._id != nearbyCell._id && cell._connections.size() < MAX_CELL_BONDS && nearbyCell._connections.size() < MAX_CELL_BONDS
                && !cell.isConnectedTo(nearbyCell._id)) {
                data.addConnection(cell._id, nearbyCell._id, cache);
            }
        }
    }
}

void DescriptionEditService::randomizeCellColors(CollectionDescription& data, std::vector<int> const& colorCodes)
{
    for (auto& creature : data._creatures) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : creature._cells) {
            cell._color = newColor;
        }
    }
    {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : data._cells) {
            cell._color = newColor;
        }
    }
}

void DescriptionEditService::randomizeGenomeColors(CollectionDescription& data, std::vector<int> const& colorCodes)
{
    for (auto& creature : data._creatures) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& gene : creature._genome._genes) {
            for (auto& node : gene._nodes) {
                node._color = newColor;
            }
        }
    }
}

void DescriptionEditService::randomizeEnergies(CollectionDescription& data, float minEnergy, float maxEnergy)
{
    for (auto& creature : data._creatures) {
        auto energy = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
        for (auto& cell : creature._cells) {
            cell._energy = energy;
        }
    }
    {
        auto energy = NumberGenerator::get().getRandomDouble(toDouble(minEnergy), toDouble(maxEnergy));
        for (auto& cell : data._cells) {
            cell._energy = energy;
        }
    }
}

void DescriptionEditService::randomizeAges(CollectionDescription& data, int minAge, int maxAge)
{
    for (auto& creature : data._creatures) {
        auto age = NumberGenerator::get().getRandomDouble(toDouble(minAge), toDouble(maxAge));
        for (auto& cell : creature._cells) {
            cell._age = age;
        }
    }
    {
        auto age = NumberGenerator::get().getRandomDouble(toDouble(minAge), toDouble(maxAge));
        for (auto& cell : data._cells) {
            cell._age = age;
        }
    }
}

void DescriptionEditService::randomizeCountdowns(CollectionDescription& data, int minValue, int maxValue)
{
    for (auto& creature : data._creatures) {
        auto countdown = NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue));
        for (auto& cell : creature._cells) {
            if (cell.getCellType() == CellType_Detonator) {
                std::get<DetonatorDescription>(cell._cellTypeData)._countdown = countdown;
            }
        }
    }
    {
        auto countdown = NumberGenerator::get().getRandomDouble(toDouble(minValue), toDouble(maxValue));
        for (auto& cell : data._cells) {
            if (cell.getCellType() == CellType_Detonator) {
                std::get<DetonatorDescription>(cell._cellTypeData)._countdown = countdown;
            }
        }
    }
}

void DescriptionEditService::randomizeMutationIds(CollectionDescription& data)
{
    for (auto& creature : data._creatures) {
        creature._mutationId = NumberGenerator::get().getRandomInt() % 0xffff;
    }
}

void DescriptionEditService::removeMetadata(CollectionDescription& data)
{
    data.forEach([&](CellDescription& cell) { removeMetadata(cell); });
}

void DescriptionEditService::assignNewObjectAndCreatureIds(CollectionDescription& data)
{
    std::unordered_map<uint64_t, uint64_t> oldToNewObjectIds;
    std::unordered_map<uint64_t, uint64_t> oldToNewCreatureIds;

    auto replaceCreatureId = [&](uint64_t& oldCreatureId) {
        auto findResult = oldToNewCreatureIds.find(oldCreatureId);
        if (findResult != oldToNewCreatureIds.end()) {
            oldCreatureId = findResult->second;
        } else {
            auto newCreatureId = NumberGenerator::get().createCreatureId();
            oldToNewCreatureIds.emplace(oldCreatureId, newCreatureId);
            oldCreatureId = newCreatureId;
        }
    };
    for (auto& creature : data._creatures) {
        replaceCreatureId(creature._id);
    }
    data.forEach([&](CellDescription& cell) {
        auto newObjectId = NumberGenerator::get().createObjectId();
        oldToNewObjectIds.emplace(cell._id, newObjectId);
        cell._id = newObjectId;
    });

    data.forEach([&](CellDescription& cell) {
        for (auto& connection : cell._connections) {
            auto findResult = oldToNewObjectIds.find(connection._cellId);
            if (findResult != oldToNewObjectIds.end()) {
                connection._cellId = findResult->second;
            }
        }
    });

    for (auto& particle : data._particles) {
        particle._id = NumberGenerator::get().createObjectId();
    }
}

void DescriptionEditService::setCenter(CollectionDescription& collection, RealVector2D const& center)
{
    auto origCenter = calcCenter(collection);
    auto delta = center - origCenter;
    shift(collection, delta);
}

RealVector2D DescriptionEditService::calcCenter(CollectionDescription const& collection) const
{
    RealVector2D result;
    auto numEntities = collection._cells.size() + collection._particles.size();
    for (auto const& creatures : collection._creatures) {
        numEntities += creatures._cells.size();
    }
    collection.forEach([&](CellDescription const& cell) { result += cell._pos; });

    for (auto const& particle : collection._particles) {
        result += particle._pos;
    }
    result /= numEntities;
    return result;
}

RealVector2D DescriptionEditService::calcCenter(CreatureDescription const& creature) const
{
    CHECK(!creature._cells.empty());

    RealVector2D result;
    for (auto const& cell : creature._cells) {
        result += cell._pos;
    }
    result /= creature._cells.size();
    return result;
}

void DescriptionEditService::shift(CollectionDescription& collection, RealVector2D const& delta)
{
    collection.forEach([&](CellDescription& cell) { cell._pos += delta; });

    for (auto& particle : collection._particles) {
        particle._pos += delta;
    }
}

void DescriptionEditService::rotate(CollectionDescription& collection, float angle)
{
    auto rotationMatrix = Math::calcRotationMatrix(angle);
    auto center = calcCenter(collection);

    auto rotate = [&](RealVector2D& pos) {
        auto relPos = pos - center;
        auto rotatedRelPos = rotationMatrix * relPos;
        pos = center + rotatedRelPos;
    };
    collection.forEach([&](CellDescription& cell) { rotate(cell._pos); });
    for (auto& particle : collection._particles) {
        rotate(particle._pos);
    }
}

void DescriptionEditService::accelerate(CollectionDescription& collection, RealVector2D const& velDelta, float angularVelDelta)
{
    auto center = calcCenter(collection);

    auto accelerate = [&](RealVector2D const& pos, RealVector2D& vel) {
        auto relPos = pos - center;
        vel += Physics::tangentialVelocity(relPos, velDelta, angularVelDelta);
    };
    collection.forEach([&](CellDescription& cell) { accelerate(cell._pos, cell._vel); });
    for (auto& particle : collection._particles) {
        accelerate(particle._pos, particle._vel);
    }
}

void DescriptionEditService::removeMetadata(CellDescription& cell)
{
    cell._metadata._description.clear();
    cell._metadata._name.clear();
}

bool DescriptionEditService::isCellPresent(Occupancy const& cellPosBySlot, SpaceCalculator const& spaceCalculator, RealVector2D const& posToCheck, float distance)
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

uint64_t DescriptionEditService::getId(ExtendedCellOrParticleDescription const& entity)
{
    if (std::holds_alternative<ExtendedCellDescription>(entity)) {
        return std::get<ExtendedCellDescription>(entity).cell._id;
    }
    return std::get<ParticleDescription>(entity)._id;
}

RealVector2D DescriptionEditService::getPos(ExtendedCellOrParticleDescription const& entity)
{
    if (std::holds_alternative<ExtendedCellDescription>(entity)) {
        return std::get<ExtendedCellDescription>(entity).cell._pos;
    }
    return std::get<ParticleDescription>(entity)._pos;
}

std::vector<ExtendedCellOrParticleDescription> DescriptionEditService::getObjects(
    CollectionDescription const& data)
{
    std::vector<ExtendedCellOrParticleDescription> result;
    for (auto const& particle : data._particles) {
        result.emplace_back(particle);
    }
    for (auto const& cell : data._cells) {
        ExtendedCellDescription extCell;
        extCell.cell = cell;
        result.emplace_back(extCell);
    }
    for (auto const& creature : data._creatures) {
        for (auto const& cell : creature._cells) {
            ExtendedCellDescription extCell;
            extCell.cell = cell;
            extCell.creatureId = creature._id;
            extCell.genome = creature._genome;
            result.emplace_back(extCell);
        }
    }
    return result;
}

namespace
{
    template <typename T1, typename T2>
    bool contains(std::vector<T1> const& a, std::vector<T2> const& b)
    {
        for (auto i = a.begin(), y = a.end(); i != y; ++i) {
            bool match = true;

            auto ii = i;
            for (auto j = b.begin(), z = b.end(); j != z; ++j) {
                if (ii == a.end() || *j != *ii) {
                    match = false;
                    break;
                }
                ii++;
            }
            if (match) {
                return true;
            }
        }
        return false;
    }
}

std::vector<ExtendedCellOrParticleDescription> DescriptionEditService::getCellsForCreatureRepresentatives(CollectionDescription const& data)
{
    return {};
}
