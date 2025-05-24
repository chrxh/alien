#include "DescriptionEditService.h"

#include <cmath>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/map.hpp>

#include "Base/NumberGenerator.h"
#include "Base/Math.h"
#include "GenomeDescriptions.h"
#include "SpaceCalculator.h"
#include "GenomeDescriptionConverterService.h"

CollectionDescription DescriptionEditService::createRect(CreateRectParameters const& parameters)
{
    CollectionDescription result;
    auto creatureId = parameters._randomCreatureId ? toInt(NumberGenerator::get().getRandomInt(std::numeric_limits<int>::max())) : 0;
    for (int i = 0; i < parameters._width; ++i) {
        for (int j = 0; j < parameters._height; ++j) {
            result.addCell(CellDescription()
                               .id(NumberGenerator::get().getId())
                               .pos({toFloat(i) * parameters._cellDistance, toFloat(j) * parameters._cellDistance})
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky)
                               .creatureId(creatureId)
                               .mutationId(parameters._mutationId)
                               .genomeComplexity(parameters._genomeComplexity)
                               .cellType(parameters._cellType));
        }
    }
    reconnectCells(result, parameters._cellDistance * 1.1f);
    result.setCenter(parameters._center);
    return result;
}

CollectionDescription DescriptionEditService::createHex(CreateHexParameters const& parameters)
{
    CollectionDescription result;
    auto creatureId = parameters._randomCreatureId ? toInt(NumberGenerator::get().getRandomInt(std::numeric_limits<int>::max())) : 0;
    auto incY = sqrt(3.0) * parameters._cellDistance / 2.0;
    for (int j = 0; j < parameters._layers; ++j) {
        for (int i = -(parameters._layers - 1); i < parameters._layers - j; ++i) {

            //create cell: upper layer
            result.addCell(CellDescription()
                               .id(NumberGenerator::get().getId())
                               .cellType(StructureCellDescription())
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(-j * incY)})
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky)
                               .creatureId(creatureId));

            //create cell: under layer (except for 0-layer)
            if (j > 0) {
                result.addCell(CellDescription()
                                 .id(NumberGenerator::get().getId())
                                   .cellType(StructureCellDescription())
                                   .energy(parameters._energy)
                                   .stiffness(parameters._stiffness)
                                   .pos({toFloat(i * parameters._cellDistance + j * parameters._cellDistance / 2.0), toFloat(j * incY)})
                                   .color(parameters._color)
                                   .barrier(parameters._barrier)
                                   .creatureId(creatureId));
            }
        }
    }

    reconnectCells(result, parameters._cellDistance * 1.5f);
    result.setCenter(parameters._center);

    return result;
}

CollectionDescription DescriptionEditService::createUnconnectedCircle(CreateUnconnectedCircleParameters const& parameters)
{
    CollectionDescription result;
    auto creatureId = parameters._randomCreatureId ? toInt(NumberGenerator::get().getRandomInt(std::numeric_limits<int>::max())) : 0;

    if (parameters._radius <= 1 + NEAR_ZERO) {
        result.addCell(CellDescription()
                           .id(NumberGenerator::get().getId())
                           .cellType(StructureCellDescription())
                           .pos(parameters._center)
                           .energy(parameters._energy)
                           .stiffness(parameters._stiffness)
                           .color(parameters._color)
                           .barrier(parameters._barrier)
                           .sticky(parameters._sticky)
                           .creatureId(creatureId));
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
                               .id(NumberGenerator::get().getId())
                               .cellType(StructureCellDescription())
                               .energy(parameters._energy)
                               .stiffness(parameters._stiffness)
                               .pos({parameters._center.x + dxMod, parameters._center.y + dy})
                               .color(parameters._color)
                               .barrier(parameters._barrier)
                               .sticky(parameters._sticky)
                               .creatureId(creatureId));

        }
    }
    return result;
}

namespace
{
    void generateNewIds(CollectionDescription& data)
    {
        auto& numberGen = NumberGenerator::get();
        std::unordered_map<uint64_t, uint64_t> newByOldIds;
        for (auto& cell : data._cells) {
            uint64_t newId = numberGen.getId();
            newByOldIds.insert_or_assign(cell._id, newId);
            cell._id = newId;
        }

        for (auto& cell : data._cells) {
            for (auto& connection : cell._connections) {
                connection._cellId = newByOldIds.at(connection._cellId);
            }
        }
    }

    void generateNewIds(ClusterDescription& cluster)
    {
        auto& numberGen = NumberGenerator::get();
        //cluster.id = numberGen.getId();
        std::unordered_map<uint64_t, uint64_t> newByOldIds;
        for (auto& cell : cluster._cells) {
            uint64_t newId = numberGen.getId();
            newByOldIds.insert_or_assign(cell._id, newId);
            cell._id = newId;
        }

        for (auto& cell : cluster._cells) {
            for (auto& connection : cell._connections) {
                connection._cellId = newByOldIds.at(connection._cellId);
            }
        }
    }
}

void DescriptionEditService::duplicate(ClusteredCollectionDescription& data, IntVector2D const& origSize, IntVector2D const& size)
{
    ClusteredCollectionDescription result;

    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            generateNewCreatureIds(data);

            for (auto cluster : data._clusters) {
                auto origPos = cluster.getClusterPosFromCells();
                RealVector2D clusterPos = {origPos.x + incX, origPos.y + incY};
                if (clusterPos.x < size.x && clusterPos.y < size.y) {
                    for (auto& cell : cluster._cells) {
                        cell._pos = RealVector2D{cell._pos.x + incX, cell._pos.y + incY};
                        if (incX > 0 || incY > 0) {
                            removeMetadata(cell);
                        }
                    }
                    generateNewIds(cluster);

                    result.addCluster(cluster);
                }
            }
            for (auto particle : data._particles) {
                auto origPos = particle._pos;
                particle._pos = RealVector2D{origPos.x + incX, origPos.y + incY};
                if (particle._pos.x < size.x && particle._pos.y < size.y) {
                    particle.id(NumberGenerator::get().getId());
                    result.addParticle(particle);
                }
            }
        }
    }
    data = result;
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
            templateData.shift({i * parameters._horizontalDistance, j * parameters._verticalDistance});
            templateData.rotate(i * parameters._horizontalAngleInc + j * parameters._verticalAngleInc);
            templateData.accelerate(
                {i * parameters._horizontalVelXinc + j * parameters._verticalVelXinc, i * parameters._horizontalVelYinc + j * parameters._verticalVelYinc},
                i * parameters._horizontalAngularVelInc + j * parameters._verticalAngularVelInc);

            generateNewIds(templateData);
            generateNewCreatureIds(templateData);
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
        int index = 0;
        for (auto const& cell : existentData._cells) {
            auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
            cellPosBySlot[intPos].emplace_back(cell._pos);
            ++index;
        }
    }

    //do multiplication
    CollectionDescription result = input;
    generateNewIds(result);
    auto& numberGen = NumberGenerator::get();
    for (int i = 0; i < parameters._number; ++i) {
        bool overlapping = false;
        CollectionDescription copy;
        int attempts = 0;
        do {
            copy = input;
            removeMetadata(copy);
            copy.shift({toFloat(numberGen.getRandomReal(0, toInt(worldSize.x))), toFloat(numberGen.getRandomReal(0, toInt(worldSize.y)))});
            copy.rotate(toInt(numberGen.getRandomReal(parameters._minAngle, parameters._maxAngle)));
            copy.accelerate(
                {toFloat(numberGen.getRandomReal(parameters._minVelX, parameters._maxVelX)),
                 toFloat(numberGen.getRandomReal(parameters._minVelY, parameters._maxVelY))},
                toFloat(numberGen.getRandomReal(parameters._minAngularVel, parameters._maxAngularVel)));

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

        generateNewIds(copy);
        generateNewCreatureIds(copy);
        result.add(copy);

        //add copy to existentData for overlapping check
        if (parameters._overlappingCheck) {
            for (auto const& cell : copy._cells) {
                auto index = toInt(existentData._cells.size());
                existentData._cells.emplace_back(cell);
                auto intPos = toIntVector2D(spaceCalculator.getCorrectedPosition(cell._pos));
                cellPosBySlot[intPos].emplace_back(cell._pos);
            }
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

    std::unordered_map<uint64_t, int> cache;
    for (auto const& [index, cell] : data._cells | boost::adaptors::indexed(0)) {
        cache.emplace(cell._id, static_cast<int>(index));
    }
    for (auto& cell : data._cells) {
        auto nearbyCellIndices = getCellIndicesWithinRadius(data, cellIndicesBySlot, cell._pos, maxDistance);
        for (auto const& nearbyCellIndex : nearbyCellIndices) {
            auto const& nearbyCell = data._cells.at(nearbyCellIndex);
            if (cell._id != nearbyCell._id && cell._connections.size() < MAX_CELL_BONDS && nearbyCell._connections.size() < MAX_CELL_BONDS
                && !cell.isConnectedTo(nearbyCell._id)) {
                data.addConnection(cell._id, nearbyCell._id, &cache);
            }
        }
    }
}

void DescriptionEditService::correctConnections(ClusteredCollectionDescription& data, IntVector2D const& worldSize)
{
    auto threshold = std::min(worldSize.x, worldSize.y) /3;
    std::unordered_map<uint64_t, CellDescription&> cellById;
    for (auto& cluster : data._clusters) {
        for (auto& cell : cluster._cells) {
            cellById.emplace(cell._id, cell);
        }
    }
    for (auto& cluster : data._clusters) {
        for (auto& cell: cluster._cells) {
            std::vector<ConnectionDescription> newConnections;
            float angleToAdd = 0;
            for (auto connection : cell._connections) {
                auto& connectingCell = cellById.at(connection._cellId);
                if (/*spaceCalculator.distance*/Math::length(cell._pos - connectingCell._pos) > threshold) {
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

void DescriptionEditService::randomizeCellColors(ClusteredCollectionDescription& data, std::vector<int> const& colorCodes)
{
    for (auto& cluster : data._clusters) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : cluster._cells) {
            cell._color = newColor;
        }
    }
}

namespace
{
    void colorizeGenomeNodes(std::vector<uint8_t>& genome, int color)
    {
        auto desc = GenomeDescriptionConverterService::get().convertBytesToDescription(genome);
        for (auto& node : desc._cells) {
            node._color = color;
            if (node.hasGenome()) {
                colorizeGenomeNodes(node.getGenomeRef(), color);
            }
        }
        genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(desc);
    }
}

void DescriptionEditService::randomizeGenomeColors(ClusteredCollectionDescription& data, std::vector<int> const& colorCodes)
{
    for (auto& cluster : data._clusters) {
        auto newColor = colorCodes[NumberGenerator::get().getRandomInt(toInt(colorCodes.size()))];
        for (auto& cell : cluster._cells) {
            if (cell.hasGenome()) {
                colorizeGenomeNodes(cell.getGenomeRef(), newColor);
            }
        }
    }
}

void DescriptionEditService::randomizeEnergies(ClusteredCollectionDescription& data, float minEnergy, float maxEnergy)
{
    for (auto& cluster : data._clusters) {
        auto energy = NumberGenerator::get().getRandomReal(toDouble(minEnergy), toDouble(maxEnergy));
        for (auto& cell : cluster._cells) {
            cell._energy = energy;
        }
    }
}

void DescriptionEditService::randomizeAges(ClusteredCollectionDescription& data, int minAge, int maxAge)
{
    for (auto& cluster : data._clusters) {
        auto age = NumberGenerator::get().getRandomReal(toDouble(minAge), toDouble(maxAge));
        for (auto& cell : cluster._cells) {
            cell._age = age;
        }
    }
}

void DescriptionEditService::randomizeCountdowns(ClusteredCollectionDescription& data, int minValue, int maxValue)
{
    for (auto& cluster : data._clusters) {
        auto countdown = NumberGenerator::get().getRandomReal(toDouble(minValue), toDouble(maxValue));
        for (auto& cell : cluster._cells) {
            if (cell.getCellType() == CellType_Detonator) {
                std::get<DetonatorDescription>(cell._cellTypeData)._countdown = countdown;
            }
        }
    }
}

void DescriptionEditService::randomizeMutationIds(ClusteredCollectionDescription& data)
{
    for (auto& cluster : data._clusters) {
        auto mutationId = NumberGenerator::get().getRandomInt() % 65536;
        for (auto& cell : cluster._cells) {
            cell._mutationId = toInt(mutationId);
            if (cell.getCellType() == CellType_Constructor) {
                std::get<ConstructorDescription>(cell._cellTypeData)._offspringMutationId = toInt(mutationId);
            }
        }
    }
}

void DescriptionEditService::removeMetadata(CollectionDescription& data)
{
    for(auto& cell : data._cells) {
        removeMetadata(cell);
    }
}

namespace
{
    int getNewCreatureId(int origCreatureId, std::unordered_map<int, int>& origToNewCreatureIdMap)
    {
        auto findResult = origToNewCreatureIdMap.find(origCreatureId);
        if (findResult != origToNewCreatureIdMap.end()) {
            return findResult->second;
        } else {
            int newCreatureId = 0;
            while (newCreatureId == 0) {
                newCreatureId = NumberGenerator::get().getRandomInt();
            }
            origToNewCreatureIdMap.emplace(origCreatureId, newCreatureId);
            return newCreatureId;
        }
    };

}

void DescriptionEditService::generateNewCreatureIds(CollectionDescription& data)
{
    std::unordered_map<int, int> origToNewCreatureIdMap;
    for (auto& cell : data._cells) {
        if (cell._creatureId != 0) {
            cell._creatureId = getNewCreatureId(cell._creatureId, origToNewCreatureIdMap);
        }
        if (cell.getCellType() == CellType_Constructor) {
            auto& offspringCreatureId = std::get<ConstructorDescription>(cell._cellTypeData)._offspringCreatureId;
            offspringCreatureId = getNewCreatureId(offspringCreatureId, origToNewCreatureIdMap);
        }
    }
}

void DescriptionEditService::generateNewCreatureIds(ClusteredCollectionDescription& data)
{
    std::unordered_map<int, int> origToNewCreatureIdMap;
    for (auto& cluster: data._clusters) {
        for (auto& cell : cluster._cells) {
            if (cell._creatureId != 0) {
                cell._creatureId = getNewCreatureId(cell._creatureId, origToNewCreatureIdMap);
            }
            if (cell.getCellType() == CellType_Constructor) {
                auto& offspringCreatureId = std::get<ConstructorDescription>(cell._cellTypeData)._offspringCreatureId;
                offspringCreatureId = getNewCreatureId(offspringCreatureId, origToNewCreatureIdMap);
            }
        }
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

uint64_t DescriptionEditService::getId(CellOrParticleDescription const& entity)
{
    if (std::holds_alternative<CellDescription>(entity)) {
        return std::get<CellDescription>(entity)._id;
    }
    return std::get<ParticleDescription>(entity)._id;
}

RealVector2D DescriptionEditService::getPos(CellOrParticleDescription const& entity)
{
    if (std::holds_alternative<CellDescription>(entity)) {
        return std::get<CellDescription>(entity)._pos;
    }
    return std::get<ParticleDescription>(entity)._pos;
}

std::vector<CellOrParticleDescription> DescriptionEditService::getObjects(
    CollectionDescription const& data)
{
    std::vector<CellOrParticleDescription> result;
    for (auto const& particle : data._particles) {
        result.emplace_back(particle);
    }
    for (auto const& cell : data._cells) {
        result.emplace_back(cell);
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

std::vector<CellOrParticleDescription> DescriptionEditService::getConstructorToMainGenomes(CollectionDescription const& data)
{
    std::map<std::vector<uint8_t>, size_t> genomeToCellIndex;
    for (auto const& [index, cell] : data._cells | boost::adaptors::indexed(0)) {
        if (cell.getCellType() == CellType_Constructor) {
            auto const& genome = std::get<ConstructorDescription>(cell._cellTypeData)._genome;
            if (!genomeToCellIndex.contains(genome) || cell._livingState != LivingState_UnderConstruction) {
                genomeToCellIndex[genome] = index;
            }
        }
    }
    std::vector<std::pair<std::vector<uint8_t>, size_t>> genomeAndCellIndex;
    for (auto const& [genome, index] : genomeToCellIndex) {
        genomeAndCellIndex.emplace_back(std::make_pair(genome, index));
    }
    std::ranges::sort(genomeAndCellIndex, [](auto const& element1, auto const& element2) { return element1.first.size() > element2.first.size(); });

    std::vector<CellOrParticleDescription> result;
    for (auto it = genomeAndCellIndex.begin(); it != genomeAndCellIndex.end(); ++it) {
        bool alreadyContained = false;
        for (auto it2 = genomeAndCellIndex.begin(); it2 != it; ++it2) {
            auto const& genome1 = it->first;
            auto const& genome2 = it2->first;
            if (contains(genome2, genome1)) {
                alreadyContained = true;
                break;
            }
        }
        if (!alreadyContained) {
            result.emplace_back(data._cells.at(it->second));
        }
    }
    return result;
}
