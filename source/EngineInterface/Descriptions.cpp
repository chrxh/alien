#include "Descriptions.h"

#include <boost/range/adaptors.hpp>

#include "Base/Math.h"
#include "Base/Physics.h"

#include "NumberGenerator.h"

NeuralNetworkDescription::NeuralNetworkDescription()
{
    _weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
    _biases.resize(MAX_CHANNELS, 0);
    _activationFunctions.resize(MAX_CHANNELS, ActivationFunction_Identity);
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        _weights[i * MAX_CHANNELS + i] = 1.0f;
    }
}

NeuralNetworkDescription& NeuralNetworkDescription::weight(int row, int col, float value)
{
    _weights[row * MAX_CHANNELS + col] = value;
    return *this;
}

ConstructorDescription::ConstructorDescription()
{
}

SignalDescription::SignalDescription()
{
    _channels.resize(MAX_CHANNELS, 0);
}

MuscleMode MuscleDescription::getMode() const
{
    if (std::holds_alternative<AutoBendingDescription>(_mode)) {
        return MuscleMode_AutoBending;
    } else if (std::holds_alternative<ManualBendingDescription>(_mode)) {
        return MuscleMode_ManualBending;
    } else if (std::holds_alternative<AngleBendingDescription>(_mode)) {
        return MuscleMode_AngleBending;
    } else if (std::holds_alternative<AutoCrawlingDescription>(_mode)) {
        return MuscleMode_AutoCrawling;
    } else if (std::holds_alternative<ManualCrawlingDescription>(_mode)) {
        return MuscleMode_ManualCrawling;
    } else if (std::holds_alternative<DirectMovementDescription>(_mode)) {
        return MuscleMode_DirectMovement;
    }
    THROW_NOT_IMPLEMENTED();
}

InjectorDescription::InjectorDescription()
{
}

CellDescription::CellDescription(bool createIds)
{
    if (createIds) {
        _id = NumberGenerator::get().createObjectId();
    }
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
    } else if (std::holds_alternative<GeneratorDescription>(_cellTypeData)) {
        return CellType_Generator;
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

bool CellDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(_connections.begin(), _connections.end(), [&id](auto const& connection) { return connection._cellId == id; }) != _connections.end();
}

CellDescription& CellDescription::signalAndRelaxTime(std::vector<float> const& value)
{
    CHECK(value.size() == MAX_CHANNELS);

    SignalDescription newSignal;
    newSignal._channels = value;
    _signal = newSignal;
    _signalRelaxationTime = MAX_SIGNAL_RELAXATION_TIME;
    return *this;
}

CellDescription& CellDescription::signalRoutingRestriction(float baseAngle, float openingAngle)
{
    SignalRoutingRestrictionDescription routingRestriction;
    routingRestriction._active = true;
    routingRestriction._baseAngle = baseAngle;
    routingRestriction._openingAngle = openingAngle;
    _signalRoutingRestriction = routingRestriction;
    return *this;
}

ClusterDescription& ClusterDescription::addCells(std::vector<CellDescription> const& value)
{
    _cells.insert(_cells.end(), value.begin(), value.end());
    return *this;
}

ClusterDescription& ClusterDescription::addCell(CellDescription const& value)
{
    addCells({value});
    return *this;
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

ParticleDescription::ParticleDescription()
{
    _id = NumberGenerator::get().createObjectId();
}

CreatureDescription::CreatureDescription()
{
    _id = NumberGenerator::get().createCreatureId();
}

void CollectionDescription::forEach(std::function<void(CellDescription const&)> const& applyFunc) const
{
    for (auto& cell : _cells) {
        applyFunc(cell);
    }
    for (auto& creature : _creatures) {
        for (auto& cell : creature._cells) {
            applyFunc(cell);
        }
    }
}

void CollectionDescription::forEach(std::function<void(CellDescription&)> const& applyFunc)
{
    for (auto& cell : _cells) {
        applyFunc(cell);
    }
    for (auto& creature : _creatures) {
        for (auto& cell : creature._cells) {
            applyFunc(cell);
        }
    }
}

CollectionDescription& CollectionDescription::add(CollectionDescription const& other)
{
    _cells.insert(_cells.end(), other._cells.begin(), other._cells.end());
    _particles.insert(_particles.end(), other._particles.begin(), other._particles.end());
    _creatures.insert(_creatures.end(), other._creatures.begin(), other._creatures.end());
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

std::unordered_set<uint64_t> CollectionDescription::getCellIds() const
{
    std::unordered_set<uint64_t> result;
    for (auto const& cell : _cells) {
        result.insert(cell._id);
    }
    return result;
}

CollectionCache CollectionDescription::createCache() const
{
    CollectionCache result = std::make_shared<_CollectionCache>();
    for (auto const& [creatureIndex, creature] : _creatures | boost::adaptors::indexed(0)) {
        for (auto const& [cellIndex, cell] : creature._cells | boost::adaptors::indexed(0)) {
            result->cellIdToIndex.emplace(cell._id, _CollectionCache::Index{.creatureIndex = toInt(creatureIndex), .cellIndex = toInt(cellIndex)});
        }
    }
    for (auto const& [cellIndex, cell] : _cells | boost::adaptors::indexed(0)) {
        result->cellIdToIndex.emplace(cell._id, _CollectionCache::Index{.creatureIndex = std::nullopt, .cellIndex = toInt(cellIndex)});
    }
    return result;
}

CollectionDescription& CollectionDescription::addConnection(uint64_t const& cellId1, uint64_t const& cellId2, CollectionCache const& cache)
{
    auto& cell2 = getCellRef(cellId2, cache);
    return addConnection(cellId1, cellId2, cell2._pos, cache);
}

CollectionDescription&
CollectionDescription::addConnection(uint64_t const& cellId1, uint64_t const& cellId2, RealVector2D const& refPosCell2, CollectionCache const& cache)
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

CellDescription const& CollectionDescription::getCellRef(uint64_t const& cellId, CollectionCache const& cache) const
{
    if (cache != nullptr) {
        auto index = getCellIndex(cellId, cache);
        if (index.creatureIndex.has_value()) {
            return _creatures.at(index.creatureIndex.value())._cells.at(index.cellIndex);
        } else {
            return _cells.at(index.cellIndex);
        }
    } else {
        for (auto& cell : _cells) {
            if (cell._id == cellId) {
                return cell;
            }
        }
        for (auto& creature : _creatures) {
            for (auto& cell : creature._cells) {
                if (cell._id == cellId) {
                    return cell;
                }
            }
        }
        CHECK(false);
    }
}

CellDescription& CollectionDescription::getCellRef(uint64_t const& cellId, CollectionCache const& cache)
{
    if (cache != nullptr) {
        auto index = getCellIndex(cellId, cache);
        if (index.creatureIndex.has_value()) {
            return _creatures.at(index.creatureIndex.value())._cells.at(index.cellIndex);
        } else {
            return _cells.at(index.cellIndex);
        }
    } else {
        for (auto& cell : _cells) {
            if (cell._id == cellId) {
                return cell;
            }            
        }
        for (auto& creature: _creatures) {
            for (auto& cell : creature._cells) {
                if (cell._id == cellId) {
                    return cell;
                }
            }
        }
        CHECK(false);
    }
}

CellDescription& CollectionDescription::getOtherCell(uint64_t id)
{
    for (auto& cell : _cells) {
        if (cell._id != id) {
            return cell;
        }
    }
    for (auto& creature : _creatures) {
        for (auto& cell : creature._cells) {
            if (cell._id != id) {
                return cell;
            }
        }
    }
    CHECK(false);
}

CellDescription& CollectionDescription::getOtherCell(std::set<uint64_t> const& ids)
{
    for (auto& cell : _cells) {
        if (!ids.contains(cell._id)) {
            return cell;
        }
    }
    for (auto& creature : _creatures) {
        for (auto& cell : creature._cells) {
            if (!ids.contains(cell._id)) {
                return cell;
            }
        }
    }
    CHECK(false);
}

bool CollectionDescription::hasConnection(uint64_t id, uint64_t otherId) const
{
    auto const& cell = getCellRef(id);
    for (auto const& connection : cell._connections) {
        if (connection._cellId == otherId) {
            return true;
        }
    }
    return false;
}

bool CollectionDescription::hasConnection(CellDescription const& cell1, CellDescription const& cell2) const
{
    return hasConnection(cell1._id, cell2._id);
}

ConnectionDescription CollectionDescription::getConnection(uint64_t id, uint64_t otherId) const
{
    auto cell = getCellRef(id);
    for (auto const& connection : cell._connections) {
        if (connection._cellId == otherId) {
            return connection;
        }
    }
    CHECK(false);
}

ConnectionDescription CollectionDescription::getConnection(CellDescription const& cell1, CellDescription const& cell2) const
{
    for (auto const& connection : cell1._connections) {
        if (connection._cellId == cell2._id) {
            return connection;
        }
    }
    CHECK(false);
}

CreatureDescription& CollectionDescription::getCreature(uint64_t id)
{
    for (auto& creature : _creatures) {
        if (creature._id == id) {
            return creature;
        }
    }
    CHECK(false);
}

CreatureDescription& CollectionDescription::getOtherCreature(uint64_t id)
{
    for (auto& creature : _creatures) {
        if (creature._id != id) {
            return creature;
        }
    }
    CHECK(false);
}

_CollectionCache::Index CollectionDescription::getCellIndex(uint64_t const& cellId, CollectionCache const& cache) const
{
    _CollectionCache::Index result;
    auto findResult = cache->cellIdToIndex.find(cellId);
    if (findResult != cache->cellIdToIndex.end()) {
        result = findResult->second;
    } else {
        auto found = false;
        for (auto const& [creatureIndex, creature] : _creatures | boost::adaptors::indexed(0)) {
            for (auto const& [cellIndex, cell] : creature._cells | boost::adaptors::indexed(0)) {
                result = _CollectionCache::Index{.creatureIndex = toInt(creatureIndex), .cellIndex = toInt(cellIndex)};
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        for (auto const& [cellIndex, cell] : _cells | boost::adaptors::indexed(0)) {
            result = _CollectionCache::Index{.creatureIndex = std::nullopt, .cellIndex = toInt(cellIndex)};
            found = true;
            break;
        }
        if (!found) {
            CHECK(false);
        }
    }
    return result;
}
