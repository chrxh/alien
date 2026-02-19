#include "Desc.h"

#include <algorithm>
#include <cmath>

#include <boost/range/adaptors.hpp>

#include <Base/Math.h>
#include <Base/Physics.h>

#include "NumberGenerator.h"

NeuralNetDesc::NeuralNetDesc()
{
    _weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        _weights[i * NEURONS_PER_CELL + i] = 1.0f;
    }

    _biases.resize(NEURONS_PER_CELL, 0);

    _activationFunctions.resize(NEURONS_PER_CELL, ActivationFunction_Identity);
    
    _connectionWeights.resize(MAX_OBJECT_CONNECTIONS, 0);
    _connectionWeights.at(0) = 1.0f;
}

NeuralNetDesc& NeuralNetDesc::weight(int row, int col, NeuralNetWeight value)
{
    _weights[row * NEURONS_PER_CELL + col] = value;
    return *this;
}

NeuralNetDesc& NeuralNetDesc::bias(int row, float value)
{
    _biases[row] = value;
    return *this;
}

NeuralNetDesc& NeuralNetDesc::connectionWeight(int connectionIndex, float value)
{
    CHECK(connectionIndex < MAX_OBJECT_CONNECTIONS);
    _connectionWeights[connectionIndex] = value;
    return *this;
}

SignalDesc::SignalDesc()
{
    _channels.resize(NEURONS_PER_CELL, 0);
}

SignalDesc& SignalDesc::channels(std::vector<float> const& value)
{
    CHECK(value.size() == NEURONS_PER_CELL);
    _channels = value;
    return *this;
}

SensorMode SensorDesc::getMode() const
{
    if (std::holds_alternative<TelemetryDesc>(_mode)) {
        return SensorMode_Telemetry;
    } else if (std::holds_alternative<DetectEnergyDesc>(_mode)) {
        return SensorMode_DetectEnergy;
    } else if (std::holds_alternative<DetectStructureDesc>(_mode)) {
        return SensorMode_DetectStructure;
    } else if (std::holds_alternative<DetectFreeCellDesc>(_mode)) {
        return SensorMode_DetectFreeCell;
    } else if (std::holds_alternative<DetectCreatureDesc>(_mode)) {
        return SensorMode_DetectCreature;
    }
    THROW_NOT_IMPLEMENTED();
}

MuscleMode MuscleDesc::getMode() const
{
    if (std::holds_alternative<AutoBendingDesc>(_mode)) {
        return MuscleMode_AutoBending;
    } else if (std::holds_alternative<ManualBendingDesc>(_mode)) {
        return MuscleMode_ManualBending;
    } else if (std::holds_alternative<AngleBendingDesc>(_mode)) {
        return MuscleMode_AngleBending;
    } else if (std::holds_alternative<AutoCrawlingDesc>(_mode)) {
        return MuscleMode_AutoCrawling;
    } else if (std::holds_alternative<ManualCrawlingDesc>(_mode)) {
        return MuscleMode_ManualCrawling;
    } else if (std::holds_alternative<DirectMovementDesc>(_mode)) {
        return MuscleMode_DirectMovement;
    }
    THROW_NOT_IMPLEMENTED();
}

ReconnectorMode ReconnectorDesc::getMode() const
{
    if (std::holds_alternative<ReconnectStructureDesc>(_mode)) {
        return ReconnectorMode_Structure;
    } else if (std::holds_alternative<ReconnectFreeCellDesc>(_mode)) {
        return ReconnectorMode_FreeCell;
    } else if (std::holds_alternative<ReconnectCreatureDesc>(_mode)) {
        return ReconnectorMode_Creature;
    }
    THROW_NOT_IMPLEMENTED();
}

GeneratorMode GeneratorDesc::getMode() const
{
    if (std::holds_alternative<SquareSignalDesc>(_mode)) {
        return GeneratorMode_SquareSignal;
    } else if (std::holds_alternative<SawtoothSignalDesc>(_mode)) {
        return GeneratorMode_SawtoothSignal;
    }
    THROW_NOT_IMPLEMENTED();
}

SignalEntryDesc::SignalEntryDesc()
{
    _channels.resize(NEURONS_PER_CELL, 0);
}

AttackerMode AttackerDesc::getMode() const
{
    if (std::holds_alternative<AttackFreeCellDesc>(_mode)) {
        return AttackerMode_FreeCell;
    } else if (std::holds_alternative<AttackCreatureDesc>(_mode)) {
        return AttackerMode_Creature;
    }
    THROW_NOT_IMPLEMENTED();
}

MemoryMode MemoryDesc::getMode() const
{
    if (std::holds_alternative<SignalDelayDesc>(_mode)) {
        return MemoryMode_SignalDelay;
    } else if (std::holds_alternative<SignalRecorderDesc>(_mode)) {
        return MemoryMode_SignalRecorder;
    } else if (std::holds_alternative<SignalStorageDesc>(_mode)) {
        return MemoryMode_SignalStorage;
    } else if (std::holds_alternative<SignalIntegratorDesc>(_mode)) {
        return MemoryMode_SignalIntegrator;
    }
    THROW_NOT_IMPLEMENTED();
}

CommunicatorMode CommunicatorDesc::getMode() const
{
    if (std::holds_alternative<SenderDesc>(_mode)) {
        return CommunicatorMode_Sender;
    } else if (std::holds_alternative<ReceiverDesc>(_mode)) {
        return CommunicatorMode_Receiver;
    }
    THROW_NOT_IMPLEMENTED();
}

InjectorDesc::InjectorDesc() {}

CellType CellDesc::getCellType() const
{
    if (std::holds_alternative<BaseDesc>(_cellType)) {
        return CellType_Base;
    } else if (std::holds_alternative<DepotDesc>(_cellType)) {
        return CellType_Depot;
    } else if (std::holds_alternative<SensorDesc>(_cellType)) {
        return CellType_Sensor;
    } else if (std::holds_alternative<GeneratorDesc>(_cellType)) {
        return CellType_Generator;
    } else if (std::holds_alternative<AttackerDesc>(_cellType)) {
        return CellType_Attacker;
    } else if (std::holds_alternative<InjectorDesc>(_cellType)) {
        return CellType_Injector;
    } else if (std::holds_alternative<MuscleDesc>(_cellType)) {
        return CellType_Muscle;
    } else if (std::holds_alternative<DefenderDesc>(_cellType)) {
        return CellType_Defender;
    } else if (std::holds_alternative<ReconnectorDesc>(_cellType)) {
        return CellType_Reconnector;
    } else if (std::holds_alternative<DetonatorDesc>(_cellType)) {
        return CellType_Detonator;
    } else if (std::holds_alternative<DigestorDesc>(_cellType)) {
        return CellType_Digestor;
    } else if (std::holds_alternative<MemoryDesc>(_cellType)) {
        return CellType_Memory;
    } else if (std::holds_alternative<CommunicatorDesc>(_cellType)) {
        return CellType_Communicator;
    }
    CHECK(false);
}

CellDesc& CellDesc::signal(std::vector<float> const& value)
{
    CHECK(value.size() == NEURONS_PER_CELL);

    SignalDesc newSignal;
    newSignal._channels = value;
    _signal = newSignal;
    return *this;
}

ObjectDesc::ObjectDesc(bool createIds)
{
    if (createIds) {
        _id = NumberGenerator::get().createId();
    }
}

ObjectDesc ObjectDesc::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

ObjectType ObjectDesc::getObjectType() const
{
    if (std::holds_alternative<StructureDesc>(_type)) {
        return ObjectType_Structure;
    } else if (std::holds_alternative<FreeCellDesc>(_type)) {
        return ObjectType_FreeCell;
    } else if (std::holds_alternative<CellDesc>(_type)) {
        return ObjectType_Cell;
    }
    CHECK(false);
}

StructureDesc& ObjectDesc::getStructureRef()
{
    return std::get<StructureDesc>(_type);
}

StructureDesc const& ObjectDesc::getStructureRef() const
{
    return std::get<StructureDesc>(_type);
}

FreeCellDesc& ObjectDesc::getFreeCellRef()
{
    return std::get<FreeCellDesc>(_type);
}

FreeCellDesc const& ObjectDesc::getFreeCellRef() const
{
    return std::get<FreeCellDesc>(_type);
}

CellDesc& ObjectDesc::getCellRef()
{
    return std::get<CellDesc>(_type);
}

CellDesc const& ObjectDesc::getCellRef() const
{
    return std::get<CellDesc>(_type);
}

bool ObjectDesc::isConnectedTo(uint64_t id) const
{
    return std::find_if(_connections.begin(), _connections.end(), [&id](auto const& connection) { return connection._objectId == id; }) != _connections.end();
}

float ObjectDesc::getAngleSpan(uint64_t connectedObjectId1, uint64_t connectedObjectId2) const
{
    std::optional<float> result;
    auto numConnections = _connections.size();
    for (int i = 0; i < numConnections * 2; ++i) {
        auto const& connection = _connections.at(i % numConnections);
        if (result.has_value()) {
            *result += connection._angleFromPrevious;
        }
        if (connection._objectId == connectedObjectId1) {
            result = 0.0f;
        }
        if (result.has_value() && connection._objectId == connectedObjectId2) {
            return result.value();
        }
    }
    CHECK(false);
}

EnergyDesc::EnergyDesc()
{
    _id = NumberGenerator::get().createId();
}

EnergyDesc EnergyDesc::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

CreatureDesc::CreatureDesc()
{
    _id = NumberGenerator::get().createId();
}

CreatureDesc CreatureDesc::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

void Desc::clear()
{
    _objects.clear();
    _energies.clear();
    _creatures.clear();
    _genomes.clear();
}

bool Desc::isEmpty() const
{
    return _objects.empty() && _energies.empty();
}

Desc& Desc::add(Desc&& other, bool assignNewIds /*= true*/)
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

bool Desc::hasUniqueIds() const
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

void Desc::assignNewIds()
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
    std::unordered_map<uint64_t, uint64_t> oldToNewObjectId;
    std::unordered_set<uint64_t> nonUniqueCellIds;

    std::unordered_map<std::optional<uint64_t>, std::map<uint64_t, uint64_t>> creatureIdToOldToNewObjectId;
    std::unordered_map<std::optional<uint64_t>, std::set<uint64_t>> creatureIdToNonUniqueCellIds;

    for (auto& [index, oldId] : indexToOldCellId) {
        auto newId = NumberGenerator::get().createId();

        auto& object = _objects.at(index);
        {
            auto insertionResult = oldToNewObjectId.insert({oldId, newId});
            if (!insertionResult.second) {
                nonUniqueCellIds.insert(oldId);
            }
        }
        {
            std::optional<uint64_t> creatureId = std::nullopt;
            if (object.getObjectType() == ObjectType_Cell) {
                creatureId = object.getCellRef()._creatureId;
            }
            auto insertionResult = creatureIdToOldToNewObjectId[creatureId].insert({oldId, newId});
            if (!insertionResult.second) {
                creatureIdToNonUniqueCellIds[creatureId].insert(oldId);
            }
        }

        object._id = newId;
    }

    // Helper for finding new objectId (uses original cellIds)
    auto findNewObjectId = [&](std::optional<uint64_t> const& creatureId, uint64_t objectId) {

        // First check in creature-specific map (always preferred when available)
        auto nonUnique = false;
        auto creatureFindResult = creatureIdToOldToNewObjectId.find(creatureId);
        if (creatureFindResult != creatureIdToOldToNewObjectId.end()) {
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
            auto findResult = oldToNewObjectId.find(objectId);
            if (findResult != oldToNewObjectId.end()) {
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
        std::optional<uint64_t> creatureId = std::nullopt;
        if (object.getObjectType() == ObjectType_Cell) {
            creatureId = object.getCellRef()._creatureId;
        }
        for (auto& connection : object._connections) {
            connection._objectId = findNewObjectId(creatureId, connection._objectId);
        }
        if (object.getObjectType() == ObjectType_Cell && object.getCellRef()._constructor.has_value()) {
            auto& constructor = object.getCellRef()._constructor.value();
            if (constructor._lastConstructedCellId.has_value()) {
                constructor._lastConstructedCellId = findNewObjectId(creatureId, constructor._lastConstructedCellId.value());
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
        if (object.getObjectType() != ObjectType_Cell) {
            continue;
        }
        CHECK(!nonUniqueCreatureIds.contains(object.getCellRef()._creatureId));
        auto findResult = oldToNewCreatureId.find(object.getCellRef()._creatureId);
        if (findResult != oldToNewCreatureId.end()) {
            object.getCellRef()._creatureId = findResult->second;
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

Desc& Desc::addCreature(std::vector<ObjectDesc> const& objects, CreatureDesc const& creature, GenomeDesc const& genome)
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

        // Only cells can belong to creatures
        CHECK(object.getObjectType() == ObjectType_Cell);
        object.getCellRef()._creatureId = creature._id;
        _objects.emplace_back(object);
    }

    return *this;
}

Desc& Desc::addObjects(std::vector<ObjectDesc> const& objects)
{
    _objects.insert(_objects.end(), objects.begin(), objects.end());
    return *this;
}

size_t Desc::getNumObjects() const
{
    return _objects.size();
}

size_t Desc::getNumObjectsWithoutCreature() const
{
    return std::count_if(_objects.begin(), _objects.end(), [](auto const& object) {
        return object.getObjectType() != ObjectType_Cell;
    });
}

std::vector<ObjectDesc> Desc::getObjectsForCreature(uint64_t creatureId) const
{
    std::vector<ObjectDesc> result;
    for (auto const& object : _objects) {
        if (object.getObjectType() != ObjectType_Cell) {
            continue;
        }
        if (object.getCellRef()._creatureId == creatureId) {
            result.push_back(object);
        }
    }
    return result;
}

DescCache Desc::createCache() const
{
    DescCache result = std::make_shared<_DescCache>();
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

Desc& Desc::addConnection(uint64_t const& objectId1, uint64_t const& objectId2, DescCache const& cache)
{
    auto& object2 = getObjectRef(objectId2, cache);
    return addConnection(objectId1, objectId2, object2._pos, cache);
}

Desc& Desc::addConnection(uint64_t const& objectId1, uint64_t const& objectId2, RealVector2D const& refPosCell2, DescCache const& cache)
{
    auto& object1 = getObjectRef(objectId1, cache);
    auto& object2 = getObjectRef(objectId2, cache);

    auto addConnection = [this,
                          &cache](ObjectDesc& object, ObjectDesc& otherObject, RealVector2D const& cellRefPos, RealVector2D const& otherObjectRefPos) {
        CHECK(object._connections.size() < MAX_OBJECT_CONNECTIONS);

        auto newAngle = Math::angleOfVector(otherObjectRefPos - cellRefPos);

        if (object._connections.empty()) {
            ConnectionDesc newConnection;
            newConnection._objectId = otherObject._id;
            newConnection._distance = toFloat(Math::length(otherObjectRefPos - cellRefPos));
            newConnection._angleFromPrevious = 360.0;
            object._connections.emplace_back(newConnection);
            return;
        }
        if (1 == object._connections.size()) {
            ConnectionDesc newConnection;
            newConnection._objectId = otherObject._id;
            newConnection._distance = toFloat(Math::length(otherObjectRefPos - cellRefPos));

            auto connectedObject = getObjectRef(object._connections.front()._objectId, cache);
            auto connectedObjectDelta = connectedObject._pos - cellRefPos;
            auto prevAngle = Math::angleOfVector(connectedObjectDelta);
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

        auto firstConnectedObject = getObjectRef(object._connections.front()._objectId, cache);
        auto firstConnectedObjectDelta = firstConnectedObject._pos - cellRefPos;
        auto angle = Math::angleOfVector(firstConnectedObjectDelta);
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

        ConnectionDesc newConnection;
        newConnection._objectId = otherObject._id;
        newConnection._distance = toFloat(Math::length(otherObjectRefPos - cellRefPos));

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

ObjectDesc const& Desc::getObjectRef(uint64_t const& objectId, DescCache const& cache) const
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

ObjectDesc& Desc::getObjectRef(uint64_t const& objectId, DescCache const& cache)
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

ObjectDesc& Desc::getOtherObjectRef(uint64_t id)
{
    return getOtherObjectRef(std::set<uint64_t>{id});
}

ObjectDesc& Desc::getOtherObjectRef(std::set<uint64_t> const& ids)
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

std::vector<ObjectDesc> Desc::getOtherObjects(std::set<uint64_t> const& ids) const
{
    std::vector<ObjectDesc> result;
    for (auto const& object : _objects) {
        if (!ids.contains(object._id)) {
            result.emplace_back(object);
        }
    }
    return result;
}

CreatureDesc const& Desc::getCreatureRef(uint64_t id, DescCache const& cache) const
{
    if (cache != nullptr) {
        auto findResult = cache->creatureIdToIndex.find(id);
        if (findResult != cache->creatureIdToIndex.end()) {
            return _creatures.at(findResult->second);
        }
    } else {
        for (auto& creature : _creatures) {
            if (creature._id == id) {
                return creature;
            }
        }
    }
    CHECK(false);
}

CreatureDesc& Desc::getCreatureRef(uint64_t id, DescCache const& cache)
{
    if (cache != nullptr) {
        auto findResult = cache->creatureIdToIndex.find(id);
        if (findResult != cache->creatureIdToIndex.end()) {
            return _creatures.at(findResult->second);
        }
    } else {
        for (auto& creature : _creatures) {
            if (creature._id == id) {
                return creature;
            }
        }
    }
    CHECK(false);
}

CreatureDesc& Desc::getOtherCreatureRef(uint64_t id)
{
    for (auto& creature : _creatures) {
        if (creature._id != id) {
            return creature;
        }
    }
    CHECK(false);
}

GenomeDesc const& Desc::getGenomeRef(uint64_t const& genomeId, DescCache const& cache) const
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

bool Desc::hasConnection(uint64_t id, uint64_t otherId) const
{
    auto const& object = getObjectRef(id);
    for (auto const& connection : object._connections) {
        if (connection._objectId == otherId) {
            return true;
        }
    }
    return false;
}

bool Desc::hasConnection(ObjectDesc const& object1, ObjectDesc const& object2) const
{
    return hasConnection(object1._id, object2._id);
}

ConnectionDesc& Desc::getConnectionRef(uint64_t id, uint64_t otherId)
{
    auto& object = getObjectRef(id);
    for (auto& connection : object._connections) {
        if (connection._objectId == otherId) {
            return connection;
        }
    }
    CHECK(false);
}

ConnectionDesc const& Desc::getConnection(ObjectDesc const& object1, ObjectDesc const& object2) const
{
    for (auto const& connection : object1._connections) {
        if (connection._objectId == object2._id) {
            return connection;
        }
    }
    CHECK(false);
}

uint64_t Desc::getObjectIndex(uint64_t const& objectId, DescCache const& cache) const
{
    auto findResult = cache->objectIdToIndex.find(objectId);
    if (findResult != cache->objectIdToIndex.end()) {
        return findResult->second;
    }
    CHECK(false);
}
