#include "Description.h"

#include <boost/range/adaptors.hpp>

#include <Base/Math.h>
#include <Base/Physics.h>

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

SignalDescription::SignalDescription()
{
    _channels.resize(MAX_CHANNELS, 0);
}

SensorMode SensorDescription::getMode() const
{
    if (std::holds_alternative<TelemetryDescription>(_mode)) {
        return SensorMode_Telemetry;
    } else if (std::holds_alternative<DetectEnergyDescription>(_mode)) {
        return SensorMode_DetectEnergy;
    } else if (std::holds_alternative<DetectStructureDescription>(_mode)) {
        return SensorMode_DetectStructure;
    } else if (std::holds_alternative<DetectFreeObjectDescription>(_mode)) {
        return SensorMode_DetectFreeCell;
    } else if (std::holds_alternative<DetectCreatureDescription>(_mode)) {
        return SensorMode_DetectCreature;
    }
    THROW_NOT_IMPLEMENTED();
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

ReconnectorMode ReconnectorDescription::getMode() const
{
    if (std::holds_alternative<ReconnectStructureDescription>(_mode)) {
        return ReconnectorMode_Structure;
    } else if (std::holds_alternative<ReconnectFreeObjectDescription>(_mode)) {
        return ReconnectorMode_FreeCell;
    } else if (std::holds_alternative<ReconnectCreatureDescription>(_mode)) {
        return ReconnectorMode_Creature;
    }
    THROW_NOT_IMPLEMENTED();
}

SignalEntryDescription::SignalEntryDescription()
{
    _channels.resize(MAX_CHANNELS, 0);
}

AttackerMode AttackerDescription::getMode() const
{
    if (std::holds_alternative<AttackFreeObjectDescription>(_mode)) {
        return AttackerMode_FreeCell;
    } else if (std::holds_alternative<AttackCreatureDescription>(_mode)) {
        return AttackerMode_Creature;
    }
    THROW_NOT_IMPLEMENTED();
}

MemoryMode MemoryDescription::getMode() const
{
    if (std::holds_alternative<SignalDelayDescription>(_mode)) {
        return MemoryMode_SignalDelay;
    } else if (std::holds_alternative<SignalRecorderDescription>(_mode)) {
        return MemoryMode_SignalRecorder;
    } else if (std::holds_alternative<SignalStorageDescription>(_mode)) {
        return MemoryMode_SignalStorage;
    } else if (std::holds_alternative<SignalIntegratorDescription>(_mode)) {
        return MemoryMode_SignalIntegrator;
    }
    THROW_NOT_IMPLEMENTED();
}

CommunicatorMode CommunicatorDescription::getMode() const
{
    if (std::holds_alternative<SenderDescription>(_mode)) {
        return CommunicatorMode_Sender;
    } else if (std::holds_alternative<ReceiverDescription>(_mode)) {
        return CommunicatorMode_Receiver;
    }
    THROW_NOT_IMPLEMENTED();
}

InjectorDescription::InjectorDescription() {}

CellType CellDescription::getCellType() const
{
    if (std::holds_alternative<StructureObjectDescription>(_cellType)) {
        return CellType_Structure;
    } else if (std::holds_alternative<FreeObjectDescription>(_cellType)) {
        return CellType_Free;
    } else if (std::holds_alternative<BaseDescription>(_cellType)) {
        return CellType_Base;
    } else if (std::holds_alternative<DepotDescription>(_cellType)) {
        return CellType_Depot;
    } else if (std::holds_alternative<ConstructorDescription>(_cellType)) {
        return CellType_Constructor;
    } else if (std::holds_alternative<SensorDescription>(_cellType)) {
        return CellType_Sensor;
    } else if (std::holds_alternative<GeneratorDescription>(_cellType)) {
        return CellType_Generator;
    } else if (std::holds_alternative<AttackerDescription>(_cellType)) {
        return CellType_Attacker;
    } else if (std::holds_alternative<InjectorDescription>(_cellType)) {
        return CellType_Injector;
    } else if (std::holds_alternative<MuscleDescription>(_cellType)) {
        return CellType_Muscle;
    } else if (std::holds_alternative<DefenderDescription>(_cellType)) {
        return CellType_Defender;
    } else if (std::holds_alternative<ReconnectorDescription>(_cellType)) {
        return CellType_Reconnector;
    } else if (std::holds_alternative<DetonatorDescription>(_cellType)) {
        return CellType_Detonator;
    } else if (std::holds_alternative<DigestorDescription>(_cellType)) {
        return CellType_Digestor;
    } else if (std::holds_alternative<MemoryDescription>(_cellType)) {
        return CellType_Memory;
    } else if (std::holds_alternative<CommunicatorDescription>(_cellType)) {
        return CellType_Communicator;
    }
    CHECK(false);
}

CellDescription& CellDescription::signalAndState(std::vector<float> const& value)
{
    CHECK(value.size() == MAX_CHANNELS);

    SignalDescription newSignal;
    newSignal._channels = value;
    _signal = newSignal;
    _signalState = SignalState_Active;
    return *this;
}

CellDescription& CellDescription::signalRestriction(float baseAngle, float openingAngle)
{
    SignalRestrictionDescription routingRestriction;
    routingRestriction._mode = SignalRestrictionMode_Active;
    routingRestriction._baseAngle = baseAngle;
    routingRestriction._openingAngle = openingAngle;
    _signalRestriction = routingRestriction;
    return *this;
}

ObjectDescription::ObjectDescription(bool createIds)
{
    if (createIds) {
        _id = NumberGenerator::get().createId();
    }
}

ObjectDescription ObjectDescription::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

CellDescription& ObjectDescription::getCellRef()
{
    return std::get<CellDescription>(_type);
}

CellDescription const& ObjectDescription::getCellRef() const
{
    return std::get<CellDescription>(_type);
}

bool ObjectDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(_connections.begin(), _connections.end(), [&id](auto const& connection) { return connection._objectId == id; }) != _connections.end();
}

float ObjectDescription::getAngleSpan(uint64_t connectedCellId1, uint64_t connectedCellId2) const
{
    std::optional<float> result;
    auto numConnections = _connections.size();
    for (int i = 0; i < numConnections * 2; ++i) {
        auto const& connection = _connections.at(i % numConnections);
        if (result.has_value()) {
            *result += connection._angleFromPrevious;
        }
        if (connection._objectId == connectedCellId1) {
            result = 0.0f;
        }
        if (result.has_value() && connection._objectId == connectedCellId2) {
            return result.value();
        }
    }
    CHECK(false);
}

EnergyDescription::EnergyDescription()
{
    _id = NumberGenerator::get().createId();
}

EnergyDescription EnergyDescription::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

CreatureDescription::CreatureDescription()
{
    _id = NumberGenerator::get().createId();
}

CreatureDescription CreatureDescription::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

void Description::clear()
{
    _objects.clear();
    _energies.clear();
    _creatures.clear();
    _genomes.clear();
}

bool Description::isEmpty() const
{
    return _objects.empty() && _energies.empty();
}

Description& Description::add(Description&& other, bool assignNewIds /*= true*/)
{
    if (assignNewIds) {
        other.assignNewIds();
    }
    _objects.insert(_objects.end(), other._objects.begin(), other._objects.end());
    _energies.insert(_energies.end(), other._energies.begin(), other._energies.end());
    _creatures.insert(_creatures.end(), other._creatures.begin(), other._creatures.end());
    _genomes.insert(_genomes.end(), other._genomes.begin(), other._genomes.end());
    return *this;
}

bool Description::hasUniqueIds() const
{
    std::unordered_set<uint64_t> cellIds;
    for (auto const& object : _objects) {
        cellIds.insert(object._id);
    }
    if (cellIds.size() != _objects.size()) {
        return false;
    }

    std::unordered_set<uint64_t> creatureIds;
    for (auto const& creature : _creatures) {
        creatureIds.insert(creature._id);
    }
    if (creatureIds.size() != _creatures.size()) {
        return false;
    }

    std::unordered_set<uint64_t> genomeIds;
    for (auto const& genome : _genomes) {
        genomeIds.insert(genome._id);
    }
    if (genomeIds.size() != _genomes.size()) {
        return false;
    }

    std::unordered_set<uint64_t> particleIds;
    for (auto const& energyParticle : _energies) {
        particleIds.insert(energyParticle._id);
    }
    if (particleIds.size() != _energies.size()) {
        return false;
    }
    return true;
}

void Description::assignNewIds()
{
    // Create (index, oldCellId) vector sorted by cell id
    std::vector<std::pair<int, uint64_t>> indexToOldCellId;
    for (int i = 0; i < toInt(_objects.size()); ++i) {
        indexToOldCellId.emplace_back(i, _objects[i]._id);
    }
    // Sort by oldCellId to preserve order (lower original IDs get lower new IDs)
    std::sort(indexToOldCellId.begin(), indexToOldCellId.end(), [](auto const& lhs, auto const& rhs) {
        return lhs.second < rhs.second;
    });

    // Generate new cellIds and create maps for fast access
    std::unordered_map<uint64_t, uint64_t> oldToNewCellId;
    std::unordered_set<uint64_t> nonUniqueCellIds;

    std::unordered_map<std::optional<uint64_t>, std::map<uint64_t, uint64_t>> creatureIdToOldToNewCellId;
    std::unordered_map<std::optional<uint64_t>, std::set<uint64_t>> creatureIdToNonUniqueCellIds;

    for (auto& [index, oldId] : indexToOldCellId) {
        auto newId = NumberGenerator::get().createId();

        auto& object = _objects.at(index);
        {
            auto insertionResult = oldToNewCellId.insert({oldId, newId});
            if (!insertionResult.second) {
                nonUniqueCellIds.insert(oldId);
            }
        }
        {
            auto insertionResult = creatureIdToOldToNewCellId[std::get<CellDescription>(object._type)._creatureId].insert({oldId, newId});
            if (!insertionResult.second) {
                creatureIdToNonUniqueCellIds[std::get<CellDescription>(object._type)._creatureId].insert(oldId);
            }
        }

        object._id = newId;
    }

    // Helper for finding new objectId (uses original cellIds)
    auto findNewCellId = [&](std::optional<uint64_t> const& creatureId, uint64_t objectId) {

        // First check in creature-specific map (always preferred when available)
        auto nonUnique = false;
        auto creatureFindResult = creatureIdToOldToNewCellId.find(creatureId);
        if (creatureFindResult != creatureIdToOldToNewCellId.end()) {
            auto& creatureNonUnique = creatureIdToNonUniqueCellIds[creatureId];
            if (!creatureNonUnique.contains(objectId)) {
                auto& oldToNewMap = creatureFindResult->second;
                auto findResult = oldToNewMap.find(objectId);
                if (findResult != oldToNewMap.end()) {
                    return findResult->second;
                }
            } else {
                nonUnique = true;
            }
        }

        // Fallback: check in global map if unique globally
        if (!nonUniqueCellIds.contains(objectId)) {
            auto findResult = oldToNewCellId.find(objectId);
            if (findResult != oldToNewCellId.end()) {
                return findResult->second;
            }
        } else {
            nonUnique = true;
        }

        CHECK(!nonUnique);

        // If neither creature-specific nor global lookup worked, keep original objectId
        // This handles the case where the referenced cell doesn't exist in the DataDescription
        return objectId;
    };

    for (auto& object : _objects) {
        for (auto& connection : object._connections) {
            connection._objectId = findNewCellId(std::get<CellDescription>(object._type)._creatureId, connection._objectId);
        }
        if (std::get<CellDescription>(object._type).getCellType() == CellType_Constructor) {
            auto& constructor = std::get<ConstructorDescription>(std::get<CellDescription>(object._type)._cellType);
            if (constructor._lastConstructedCellId.has_value()) {
                constructor._lastConstructedCellId = findNewCellId(std::get<CellDescription>(object._type)._creatureId, constructor._lastConstructedCellId.value());
            }
        }
    }

    // Generate new creatureIds
    std::unordered_map<uint64_t, uint64_t> oldToNewCreatureId;
    std::unordered_set<uint64_t> nonUniqueCreatureIds;
    for (auto& creature : _creatures) {
        auto newId = NumberGenerator::get().createId();
        auto insertionResult = oldToNewCreatureId.insert({creature._id, newId});
        if (!insertionResult.second) {
            nonUniqueCreatureIds.insert(creature._id);
        }
        creature._id = newId;
    }

    // Assign new creatureIds to cells
    for (auto& object : _objects) {
        if (std::get<CellDescription>(object._type)._creatureId.has_value()) {
            CHECK(!nonUniqueCreatureIds.contains(std::get<CellDescription>(object._type)._creatureId.value()));
            auto findResult = oldToNewCreatureId.find(std::get<CellDescription>(object._type)._creatureId.value());
            if (findResult != oldToNewCreatureId.end()) {
                std::get<CellDescription>(object._type)._creatureId = findResult->second;
            }
        }
    }

    // Assign new creatureIds (ancestorId)
    for (auto& creature : _creatures) {
        if (creature._ancestorId.has_value()) {
            if (nonUniqueCreatureIds.contains(creature._ancestorId.value())) {
                creature._ancestorId.reset();
            } else {
                if (oldToNewCreatureId.contains(creature._ancestorId.value())) {
                    creature._ancestorId = oldToNewCreatureId.at(creature._ancestorId.value());
                }
            }
        }
    }

    // Generate new genomeIds
    std::unordered_map<uint64_t, uint64_t> oldToNewGenomeId;
    std::unordered_set<uint64_t> nonUniqueGenomeIds;
    for (auto& genome : _genomes) {
        auto newId = NumberGenerator::get().createId();
        auto insertionResult = oldToNewGenomeId.insert({genome._id, newId});
        if (!insertionResult.second) {
            nonUniqueGenomeIds.insert(genome._id);
        }
        genome._id = newId;
    }

    // Assign new genomeIds to creatures
    for (auto& creature : _creatures) {
        CHECK(!nonUniqueGenomeIds.contains(creature._genomeId));
        CHECK(oldToNewGenomeId.contains(creature._genomeId));
        creature._genomeId = oldToNewGenomeId.at(creature._genomeId);
    }

    // Assign new particle ids
    for (auto& energyParticle : _energies) {
        energyParticle._id = NumberGenerator::get().createId();
    }
}

Description& Description::addCreature(std::vector<ObjectDescription> const& objects, CreatureDescription const& creature, GenomeDescription const& genome)
{
    // Add genome to genomes array if not already present
    auto genomeIt = std::find_if(_genomes.begin(), _genomes.end(), [&genome](auto const& g) { return g._id == genome._id; });
    if (genomeIt == _genomes.end()) {
        _genomes.emplace_back(genome);
    }

    // Add creature with genomeId and numCells set
    auto newCreature = creature;
    newCreature._genomeId = genome._id;
    newCreature._numObjects = toInt(objects.size());
    _creatures.emplace_back(newCreature);

    // Add cells with creatureId set
    for (auto object : objects) {
        std::get<CellDescription>(object._type)._creatureId = creature._id;
        _objects.emplace_back(object);
    }

    return *this;
}

size_t Description::getNumObjects() const
{
    return _objects.size();
}

size_t Description::getNumObjectsWithoutCreature() const
{
    return std::count_if(_objects.begin(), _objects.end(), [](auto const& object) { return !std::get<CellDescription>(object._type)._creatureId.has_value(); });
}

std::vector<ObjectDescription> Description::getObjectsForCreature(uint64_t creatureId) const
{
    std::vector<ObjectDescription> result;
    for (auto const& object : _objects) {
        if (std::get<CellDescription>(object._type)._creatureId.has_value() && std::get<CellDescription>(object._type)._creatureId.value() == creatureId) {
            result.push_back(object);
        }
    }
    return result;
}

DescriptionCache Description::createCache() const
{
    DescriptionCache result = std::make_shared<_DescriptionCache>();
    for (auto const& [objectIndex, object] : _objects | boost::adaptors::indexed(0)) {
        result->objectIdToIndex.emplace(object._id, toInt(objectIndex));
    }
    for (auto const& [creatureIndex, creature] : _creatures | boost::adaptors::indexed(0)) {
        result->creatureIdToIndex.emplace(creature._id, toInt(creatureIndex));
    }
    for (auto const& [genomeIndex, genome] : _genomes | boost::adaptors::indexed(0)) {
        result->genomeIdToIndex.emplace(genome._id, genomeIndex);
    }
    return result;
}

Description& Description::addConnection(uint64_t const& objectId1, uint64_t const& objectId2, DescriptionCache const& cache)
{
    auto& object2 = getObjectRef(objectId2, cache);
    return addConnection(objectId1, objectId2, object2._pos, cache);
}

Description& Description::addConnection(uint64_t const& objectId1, uint64_t const& objectId2, RealVector2D const& refPosCell2, DescriptionCache const& cache)
{
    auto& object1 = getObjectRef(objectId1, cache);
    auto& object2 = getObjectRef(objectId2, cache);

    auto addConnection = [this,
                          &cache](ObjectDescription& object, ObjectDescription& otherCell, RealVector2D const& cellRefPos, RealVector2D const& otherCellRefPos) {
        CHECK(object._connections.size() < MAX_CELL_BONDS);

        auto newAngle = Math::angleOfVector(otherCellRefPos - cellRefPos);

        if (object._connections.empty()) {
            ConnectionDescription newConnection;
            newConnection._objectId = otherCell._id;
            newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));
            newConnection._angleFromPrevious = 360.0;
            object._connections.emplace_back(newConnection);
            return;
        }
        if (1 == object._connections.size()) {
            ConnectionDescription newConnection;
            newConnection._objectId = otherCell._id;
            newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

            auto connectedCell = getObjectRef(object._connections.front()._objectId, cache);
            auto connectedCellDelta = connectedCell._pos - cellRefPos;
            auto prevAngle = Math::angleOfVector(connectedCellDelta);
            auto angleDiff = newAngle - prevAngle;
            if (angleDiff >= 0) {
                newConnection._angleFromPrevious = toFloat(angleDiff);
                object._connections.begin()->_angleFromPrevious = 360.0f - toFloat(angleDiff);
            } else {
                newConnection._angleFromPrevious = 360.0f + toFloat(angleDiff);
                object._connections.begin()->_angleFromPrevious = toFloat(-angleDiff);
            }
            object._connections.emplace_back(newConnection);
            return;
        }

        auto firstConnectedCell = getObjectRef(object._connections.front()._objectId, cache);
        auto firstConnectedCellDelta = firstConnectedCell._pos - cellRefPos;
        auto angle = Math::angleOfVector(firstConnectedCellDelta);
        auto connectionIt = ++object._connections.begin();
        while (true) {
            auto nextAngle = angle + connectionIt->_angleFromPrevious;

            if ((angle < newAngle && newAngle <= nextAngle) || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
                break;
            }

            ++connectionIt;
            if (connectionIt == object._connections.end()) {
                connectionIt = object._connections.begin();
            }
            angle = nextAngle;
            if (angle > 360.0f) {
                angle -= 360.0f;
            }
        }

        ConnectionDescription newConnection;
        newConnection._objectId = otherCell._id;
        newConnection._distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

        auto angleDiff1 = newAngle - angle;
        if (angleDiff1 < 0) {
            angleDiff1 += 360.0f;
        }
        auto angleDiff2 = connectionIt->_angleFromPrevious;
        if (connectionIt == object._connections.begin()) {
            connectionIt = object._connections.end();  // connection at index 0 should be an invariant
        }

        auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
        newConnection._angleFromPrevious = toFloat(angleDiff2 * factor);
        connectionIt = object._connections.insert(connectionIt, newConnection);
        ++connectionIt;
        if (connectionIt == object._connections.end()) {
            connectionIt = object._connections.begin();
        }
        connectionIt->_angleFromPrevious = toFloat(angleDiff2 * (1 - factor));
    };

    addConnection(object1, object2, object1._pos, refPosCell2);
    addConnection(object2, object1, refPosCell2, object1._pos);

    return *this;
}

ObjectDescription const& Description::getObjectRef(uint64_t const& objectId, DescriptionCache const& cache) const
{
    if (cache != nullptr) {
        auto index = getObjectIndex(objectId, cache);
        return _objects.at(index);
    } else {
        for (auto const& object : _objects) {
            if (object._id == objectId) {
                return object;
            }
        }
        CHECK(false);
    }
}

ObjectDescription& Description::getObjectRef(uint64_t const& objectId, DescriptionCache const& cache)
{
    if (cache != nullptr) {
        auto index = getObjectIndex(objectId, cache);
        return _objects.at(index);
    } else {
        for (auto& object : _objects) {
            if (object._id == objectId) {
                return object;
            }
        }
        CHECK(false);
    }
}

ObjectDescription& Description::getOtherObjectRef(uint64_t id)
{
    return getOtherObjectRef(std::set<uint64_t>{id});
}

ObjectDescription& Description::getOtherObjectRef(std::set<uint64_t> const& ids)
{
    std::vector<uint64_t> matchingCells;
    for (auto& object : _objects) {
        if (!ids.contains(object._id)) {
            matchingCells.emplace_back(object._id);
        }
    }
    CHECK(matchingCells.size() == 1);

    for (auto& object : _objects) {
        if (!ids.contains(object._id)) {
            return object;
        }
    }
    CHECK(false);
}

std::vector<ObjectDescription> Description::getOtherObjects(std::set<uint64_t> const& ids) const
{
    std::vector<ObjectDescription> result;
    for (auto const& object : _objects) {
        if (!ids.contains(object._id)) {
            result.emplace_back(object);
        }
    }
    return result;
}

GenomeDescription const& Description::getGenomeRef(uint64_t const& genomeId, DescriptionCache const& cache) const
{
    if (cache != nullptr) {
        auto index = cache->genomeIdToIndex.at(genomeId);
        return _genomes.at(index);
    } else {
        for (auto& genome : _genomes) {
            if (genome._id == genomeId) {
                return genome;
            }
        }
        CHECK(false);
    }
}

bool Description::hasConnection(uint64_t id, uint64_t otherId) const
{
    auto const& object = getObjectRef(id);
    for (auto const& connection : object._connections) {
        if (connection._objectId == otherId) {
            return true;
        }
    }
    return false;
}

bool Description::hasConnection(ObjectDescription const& object1, ObjectDescription const& object2) const
{
    return hasConnection(object1._id, object2._id);
}

ConnectionDescription& Description::getConnectionRef(uint64_t id, uint64_t otherId)
{
    auto& object = getObjectRef(id);
    for (auto& connection : object._connections) {
        if (connection._objectId == otherId) {
            return connection;
        }
    }
    CHECK(false);
}

ConnectionDescription const& Description::getConnection(ObjectDescription const& object1, ObjectDescription const& object2) const
{
    for (auto const& connection : object1._connections) {
        if (connection._objectId == object2._id) {
            return connection;
        }
    }
    CHECK(false);
}

CreatureDescription& Description::getCreatureRef(uint64_t id)
{
    for (auto& creature : _creatures) {
        if (creature._id == id) {
            return creature;
        }
    }
    CHECK(false);
}

CreatureDescription& Description::getOtherCreatureRef(uint64_t id)
{
    for (auto& creature : _creatures) {
        if (creature._id != id) {
            return creature;
        }
    }
    CHECK(false);
}

uint64_t Description::getObjectIndex(uint64_t const& objectId, DescriptionCache const& cache) const
{
    auto findResult = cache->objectIdToIndex.find(objectId);
    if (findResult != cache->objectIdToIndex.end()) {
        return findResult->second;
    }
    CHECK(false);
}
