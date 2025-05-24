#include "Descriptions.h"

#include <boost/range/adaptors.hpp>

#include "GenomeDescriptionConverterService.h"
#include "Base/Math.h"
#include "Base/Physics.h"

ConstructorDescription::ConstructorDescription()
{
    _genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription());
}

InjectorDescription::InjectorDescription()
{
    _genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription());
}

CellType CellDescription::getCellType() const
{
    if (std::holds_alternative<StructureCellDescription>(_cellTypeData)) {
        return CellType_Structure;
    } else if (std::holds_alternative<FreeCellDescription>(_cellTypeData)) {
        return CellType_Free;
    } else if (std::holds_alternative<BaseDescription>(_cellTypeData)) {
        return CellType_Base;
    } else if (std::holds_alternative<DepotDescription>(_cellTypeData)) {
        return CellType_Depot;
    } else if (std::holds_alternative<ConstructorDescription>(_cellTypeData)) {
        return CellType_Constructor;
    } else if (std::holds_alternative<SensorDescription>(_cellTypeData)) {
        return CellType_Sensor;
    } else if (std::holds_alternative<OscillatorDescription>(_cellTypeData)) {
        return CellType_Oscillator;
    } else if (std::holds_alternative<AttackerDescription>(_cellTypeData)) {
        return CellType_Attacker;
    } else if (std::holds_alternative<InjectorDescription>(_cellTypeData)) {
        return CellType_Injector;
    } else if (std::holds_alternative<MuscleDescription>(_cellTypeData)) {
        return CellType_Muscle;
    } else if (std::holds_alternative<DefenderDescription>(_cellTypeData)) {
        return CellType_Defender;
    } else if (std::holds_alternative<ReconnectorDescription>(_cellTypeData)) {
        return CellType_Reconnector;
    } else if (std::holds_alternative<DetonatorDescription>(_cellTypeData)) {
        return CellType_Detonator;
    }
    CHECK(false);
}

bool CellDescription::DEPRECATED_hasGenome() const
{
    auto cellTypeEnum = getCellType();
    if (cellTypeEnum == CellType_Constructor || cellTypeEnum == CellType_Injector) {
        return true;
    }
    return false;
}

std::vector<uint8_t>& CellDescription::getGenomeRef()
{
    auto cellTypeEnum = getCellType();
    if (cellTypeEnum == CellType_Constructor) {
        return std::get<ConstructorDescription>(_cellTypeData)._genome;
    }
    if (cellTypeEnum == CellType_Injector) {
        return std::get<InjectorDescription>(_cellTypeData)._genome;
    }
    THROW_NOT_IMPLEMENTED();
}

bool CellDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(_connections.begin(), _connections.end(), [&id](auto const& connection) { return connection._cellId == id; }) != _connections.end();
}

RealVector2D ClusterDescription::getClusterPosFromCells() const
{
    RealVector2D result;
    for (auto const& cell : _cells) {
        result += cell._pos;
    }
    result /= _cells.size();
    return result;
}

void ClusteredCollectionDescription::setCenter(RealVector2D const& center)
{
    auto origCenter = calcCenter();
    auto delta = center - origCenter;
    shift(delta);
}

RealVector2D ClusteredCollectionDescription::calcCenter() const
{
    RealVector2D result;
    int numEntities = 0;
    for (auto const& cluster : _clusters) {
        for (auto const& cell : cluster._cells) {
            result += cell._pos;
            ++numEntities;
        }
    }
    for (auto const& particle : _particles) {
        result += particle._pos;
        ++numEntities;
    }
    result /= numEntities;
    return result;
}

void ClusteredCollectionDescription::shift(RealVector2D const& delta)
{
    for (auto& cluster : _clusters) {
        for (auto& cell : cluster._cells) {
            cell._pos += delta;
        }
    }
    for (auto& particle : _particles) {
        particle._pos += delta;
    }
}

int ClusteredCollectionDescription::getNumberOfCellAndParticles() const
{
    int result = static_cast<int>(_particles.size());
    for (auto const& cluster : _clusters) {
        result += static_cast<int>(cluster._cells.size());
    }
    return result;
}

CollectionDescription::CollectionDescription(ClusteredCollectionDescription const& clusteredData)
{
    for (auto const& cluster : clusteredData._clusters) {
        addCells(cluster._cells);
    }
    _particles = clusteredData._particles;
}

CollectionDescription& CollectionDescription::add(CollectionDescription const& other)
{
    _cells.insert(_cells.end(), other._cells.begin(), other._cells.end());
    _particles.insert(_particles.end(), other._particles.begin(), other._particles.end());
    return *this;
}

CollectionDescription& CollectionDescription::addCells(std::vector<CellDescription> const& value)
{
    _cells.insert(_cells.end(), value.begin(), value.end());
    return *this;
}

CollectionDescription& CollectionDescription::addCell(CellDescription const& value)
{
    addCells({value});
    return *this;
}

CollectionDescription& CollectionDescription::addParticles(std::vector<ParticleDescription> const& value)
{
    _particles.insert(_particles.end(), value.begin(), value.end());
    return *this;
}

CollectionDescription& CollectionDescription::addParticle(ParticleDescription const& value)
{
    addParticles({value});
    return *this;
}

CollectionDescription& CollectionDescription::addCreature(GenomeDescription_New const& genome, std::vector<CellDescription> const& cells)
{
    auto highestGenomeId = 0ull;
    for (auto const& cell : _cells) {
        if (cell._genomeId.has_value()) {
            highestGenomeId = std::max(highestGenomeId, cell._genomeId.value());
        }
    }

    auto newGenomeId = highestGenomeId + 1;
    auto& newGenome = _genomes.emplace_back(genome);
    newGenome._id = newGenomeId;

    auto originalSize = _cells.size();
    _cells.insert(_cells.end(), cells.begin(), cells.end());

    for (auto i = originalSize; i < _cells.size(); ++i) {
        _cells[i]._genomeId = newGenomeId;
    }

    return *this;
}

void CollectionDescription::clear()
{
    _cells.clear();
    _particles.clear();
}

bool CollectionDescription::isEmpty() const
{
    if (!_cells.empty()) {
        return false;
    }
    if (!_particles.empty()) {
        return false;
    }
    return true;
}

void CollectionDescription::setCenter(RealVector2D const& center)
{
    auto origCenter = calcCenter();
    auto delta = center - origCenter;
    shift(delta);
}

RealVector2D CollectionDescription::calcCenter() const
{
    RealVector2D result;
    auto numEntities = _cells.size() + _particles.size();
    for (auto const& cell : _cells) {
        result += cell._pos;
    }
    for (auto const& particle : _particles) {
        result += particle._pos;
    }
    result /= numEntities;
    return result;
}

void CollectionDescription::shift(RealVector2D const& delta)
{
    for (auto& cell : _cells) {
        cell._pos += delta;
    }
    for (auto& particle : _particles) {
        particle._pos += delta;
    }
}

void CollectionDescription::rotate(float angle)
{
    auto rotationMatrix = Math::calcRotationMatrix(angle);
    auto center = calcCenter();

    auto rotate = [&](RealVector2D& pos) {
        auto relPos = pos - center;
        auto rotatedRelPos = rotationMatrix * relPos;
        pos = center + rotatedRelPos;
    };
    for (auto& cell : _cells) {
        rotate(cell._pos);
    }
    for (auto& particle : _particles) {
        rotate(particle._pos);
    }
}

void CollectionDescription::accelerate(RealVector2D const& velDelta, float angularVelDelta)
{
    auto center = calcCenter();

    auto accelerate = [&](RealVector2D const& pos, RealVector2D& vel) {
        auto relPos = pos - center;
        vel += Physics::tangentialVelocity(relPos, velDelta, angularVelDelta);
    };
    for (auto& cell : _cells) {
        accelerate(cell._pos, cell._vel);
    }
    for (auto& particle : _particles) {
        accelerate(particle._pos, particle._vel);
    }
}

std::unordered_set<uint64_t> CollectionDescription::getCellIds() const
{
    std::unordered_set<uint64_t> result;
    for (auto const& cell : _cells) {
        result.insert(cell._id);
    }
    return result;
}

CollectionDescription&
CollectionDescription::addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>* cache)
{
    auto& cell2 = getCellRef(cellId2, cache);
    return addConnection(cellId1, cellId2, cell2._pos, cache);
}

CollectionDescription& CollectionDescription::addConnection(
    uint64_t const& cellId1,
    uint64_t const& cellId2,
    RealVector2D const& refPosCell2,
    std::unordered_map<uint64_t, int>* cache /*= nullptr*/)
{
    auto& cell1 = getCellRef(cellId1, cache);
    auto& cell2 = getCellRef(cellId2, cache);

    auto addConnection = [this,
                          &cache](CellDescription& cell, CellDescription& otherCell, RealVector2D const& cellRefPos, RealVector2D const& otherCellRefPos) {
        CHECK(cell._connections.size() < MAX_CELL_BONDS);

        auto newAngle = Math::angleOfVector(otherCellRefPos - cellRefPos);

        if (cell._connections.empty()) {
            ConnectionDescription newConnection;
            newConnection._cellId = otherCell._id;
            newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));
            newConnection._angleFromPrevious = 360.0;
            cell._connections.emplace_back(newConnection);
            return;
        }
        if (1 == cell._connections.size()) {
            ConnectionDescription newConnection;
            newConnection._cellId = otherCell._id;
            newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

            auto connectedCell = getCellRef(cell._connections.front()._cellId, cache);
            auto connectedCellDelta = connectedCell._pos - cellRefPos;
            auto prevAngle = Math::angleOfVector(connectedCellDelta);
            auto angleDiff = newAngle - prevAngle;
            if (angleDiff >= 0) {
                newConnection._angleFromPrevious = toFloat(angleDiff);
                cell._connections.begin()->_angleFromPrevious = 360.0f - toFloat(angleDiff);
            } else {
                newConnection._angleFromPrevious = 360.0f + toFloat(angleDiff);
                cell._connections.begin()->_angleFromPrevious = toFloat(-angleDiff);
            }
            cell._connections.emplace_back(newConnection);
            return;
        }

        auto firstConnectedCell = getCellRef(cell._connections.front()._cellId, cache);
        auto firstConnectedCellDelta = firstConnectedCell._pos - cellRefPos;
        auto angle = Math::angleOfVector(firstConnectedCellDelta);
        auto connectionIt = ++cell._connections.begin();
        while (true) {
            auto nextAngle = angle + connectionIt->_angleFromPrevious;

            if ((angle < newAngle && newAngle <= nextAngle) || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
                break;
            }

            ++connectionIt;
            if (connectionIt == cell._connections.end()) {
                connectionIt = cell._connections.begin();
            }
            angle = nextAngle;
            if (angle > 360.0f) {
                angle -= 360.0f;
            }
        }

        ConnectionDescription newConnection;
        newConnection._cellId = otherCell._id;
        newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

        auto angleDiff1 = newAngle - angle;
        if (angleDiff1 < 0) {
            angleDiff1 += 360.0f;
        }
        auto angleDiff2 = connectionIt->_angleFromPrevious;
        if (connectionIt == cell._connections.begin()) {
            connectionIt = cell._connections.end();  // connection at index 0 should be an invariant
        }

        auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
        newConnection._angleFromPrevious = toFloat(angleDiff2 * factor);
        connectionIt = cell._connections.insert(connectionIt, newConnection);
        ++connectionIt;
        if (connectionIt == cell._connections.end()) {
            connectionIt = cell._connections.begin();
        }
        connectionIt->_angleFromPrevious = toFloat(angleDiff2 * (1 - factor));
    };

    addConnection(cell1, cell2, cell1._pos, refPosCell2);
    addConnection(cell2, cell1, refPosCell2, cell1._pos);

    return *this;
}

CellDescription& CollectionDescription::getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache)
{
    if (cache) {
        auto findResult = cache->find(cellId);
        if (findResult != cache->end()) {
            return _cells.at(findResult->second);
        }
    }
    for (int i = 0; i < _cells.size(); ++i) {
        auto& cell = _cells.at(i);
        if (cell._id == cellId) {
            if (cache) {
                cache->emplace(cellId, i);
            }
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}
