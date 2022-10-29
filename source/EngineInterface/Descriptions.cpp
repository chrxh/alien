#include "Descriptions.h"

#include <boost/range/adaptors.hpp>

#include "Base/Math.h"
#include "Base/Physics.h"

namespace
{
    uint8_t convertAngleToByte(float value) { return static_cast<uint8_t>(static_cast<int8_t>(value / 180 * 128)); }
    uint8_t convertNeuronPropertyToByte(float value)
    {
        CHECK(std::abs(value) <= 2);
        return static_cast<uint8_t>(static_cast<int8_t>(value / 2 * 128));
    }
    uint8_t convertBoolToByte(bool value) { return value ? 1 : 0; }
    std::vector<uint8_t> convertWordToBytes(int value) { return {static_cast<uint8_t>(value & 0xff), static_cast<uint8_t>((value >> 8) % 0xff)}; }
}

ConstructorDescription& ConstructorDescription::setGenome(std::vector<CellDescription> const& cells, float initialAngle)
{
    genome.reserve(cells.size() * 6);
    for (auto const& [index, cell] : cells | boost::adaptors::indexed(0)) {
        float angle;
        if (index == 0) {
            angle = initialAngle;
        }
        genome.emplace_back(static_cast<uint8_t>(cell.getCellFunctionType()));
        genome.emplace_back(convertAngleToByte(angle));  //angle
        genome.emplace_back(0);     //distance
        genome.emplace_back(cell.maxConnections);
        genome.emplace_back(cell.executionOrderNumber);
        genome.emplace_back(cell.color);
        switch (cell.getCellFunctionType()) {
        case Enums::CellFunction_Neuron: {
            auto neuron = std::get<NeuronDescription>(*cell.cellFunction);
            for (int row = 0; row < MAX_CHANNELS; ++row) {
                for (int col = 0; col < MAX_CHANNELS; ++col) {
                    genome.emplace_back(convertNeuronPropertyToByte(neuron.weights[row][col]));
                }
            }
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                genome.emplace_back(convertNeuronPropertyToByte(neuron.bias[i]));
            }
        } break;
        case Enums::CellFunction_Transmitter: {
        } break;
        case Enums::CellFunction_Constructor: {
            auto constructor = std::get<ConstructorDescription>(*cell.cellFunction);
            genome.emplace_back(static_cast<uint8_t>(constructor.mode));
            genome.emplace_back(convertBoolToByte(constructor.singleConstruction));
            genome.emplace_back(convertBoolToByte(constructor.separateConstruction));
            genome.emplace_back(convertBoolToByte(constructor.makeSticky));
            genome.emplace_back(static_cast<uint8_t>(constructor.angleAlignment));
            auto makeGenomeCopy = constructor.genome.size() == 0;
            genome.emplace_back(convertBoolToByte(makeGenomeCopy));
            if (!makeGenomeCopy) {
                auto lengthBytes = convertWordToBytes(static_cast<int>(constructor.genome.size()));
                genome.insert(genome.end(), lengthBytes.begin(), lengthBytes.end());
                genome.insert(genome.end(), constructor.genome.begin(), constructor.genome.end());
            }
        } break;
        case Enums::CellFunction_Sensor: {
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
        } break;
        case Enums::CellFunction_Injector: {
        } break;
        case Enums::CellFunction_Muscle: {
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
        }
    }
    return *this;
}

Enums::CellFunction CellDescription::getCellFunctionType() const
{
    if (!cellFunction) {
        return Enums::CellFunction_None;
    }
    if (std::holds_alternative<NeuronDescription>(*cellFunction)) {
        return Enums::CellFunction_Neuron;
    }
    if (std::holds_alternative<TransmitterDescription>(*cellFunction)) {
        return Enums::CellFunction_Transmitter;
    }
    if (std::holds_alternative<ConstructorDescription>(*cellFunction)) {
        return Enums::CellFunction_Constructor;
    }
    if (std::holds_alternative<SensorDescription>(*cellFunction)) {
        return Enums::CellFunction_Sensor;
    }
    if (std::holds_alternative<NerveDescription>(*cellFunction)) {
        return Enums::CellFunction_Nerve;
    }
    if (std::holds_alternative<AttackerDescription>(*cellFunction)) {
        return Enums::CellFunction_Attacker;
    }
    if (std::holds_alternative<InjectorDescription>(*cellFunction)) {
        return Enums::CellFunction_Injector;
    }
    if (std::holds_alternative<MuscleDescription>(*cellFunction)) {
        return Enums::CellFunction_Muscle;
    }
    return Enums::CellFunction_None;
}

bool CellDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(connections.begin(), connections.end(), [&id](auto const& connection) { return connection.cellId == id; }) != connections.end();
}

RealVector2D ClusterDescription::getClusterPosFromCells() const
{
    RealVector2D result;
    for (auto const& cell : cells) {
        result += cell.pos;
    }
    result /= cells.size();
    return result;
}

void ClusteredDataDescription::setCenter(RealVector2D const& center)
{
    auto origCenter = calcCenter();
    auto delta = center - origCenter;
    shift(delta);
}

RealVector2D ClusteredDataDescription::calcCenter() const
{
    RealVector2D result;
    int numEntities = 0;
    for (auto const& cluster : clusters) {
        for (auto const& cell : cluster.cells) {
            result += cell.pos;
            ++numEntities;
        }
    }
    for (auto const& particle : particles) {
        result += particle.pos;
        ++numEntities;
    }
    result /= numEntities;
    return result;
}

void ClusteredDataDescription::shift(RealVector2D const& delta)
{
    for (auto& cluster : clusters) {
        for (auto& cell : cluster.cells) {
            cell.pos += delta;
        }
    }
    for (auto& particle : particles) {
        particle.pos += delta;
    }
}

int ClusteredDataDescription::getNumberOfCellAndParticles() const
{
    int result = static_cast<int>(particles.size());
    for (auto const& cluster : clusters) {
        result += static_cast<int>(cluster.cells.size());
    }
    return result;
}

DataDescription::DataDescription(ClusteredDataDescription const& clusteredData)
{
    for (auto const& cluster : clusteredData.clusters) {
        addCells(cluster.cells);
    }
    particles = clusteredData.particles;
}

DataDescription& DataDescription::add(DataDescription const& other)
{
    cells.insert(cells.end(), other.cells.begin(), other.cells.end());
    particles.insert(particles.end(), other.particles.begin(), other.particles.end());
    return *this;
}

DataDescription& DataDescription::addCells(std::vector<CellDescription> const& value)
{
    cells.insert(cells.end(), value.begin(), value.end());
    return *this;
}

DataDescription& DataDescription::addCell(CellDescription const& value)
{
    addCells({value});
    return *this;
}

DataDescription& DataDescription::addParticles(std::vector<ParticleDescription> const& value)
{
    particles.insert(particles.end(), value.begin(), value.end());
    return *this;
}

DataDescription& DataDescription::addParticle(ParticleDescription const& value)
{
    addParticles({value});
    return *this;
}

void DataDescription::clear()
{
    cells.clear();
    particles.clear();
}

bool DataDescription::isEmpty() const
{
    if (!cells.empty()) {
        return false;
    }
    if (!particles.empty()) {
        return false;
    }
    return true;
}

void DataDescription::setCenter(RealVector2D const& center)
{
    auto origCenter = calcCenter();
    auto delta = center - origCenter;
    shift(delta);
}

RealVector2D DataDescription::calcCenter() const
{
    RealVector2D result;
    auto numEntities = cells.size() + particles.size();
    for (auto const& cell : cells) {
        result += cell.pos;
    }
    for (auto const& particle : particles) {
        result += particle.pos;
    }
    result /= numEntities;
    return result;
}

void DataDescription::shift(RealVector2D const& delta)
{
    for (auto& cell : cells) {
        cell.pos += delta;
    }
    for (auto& particle : particles) {
        particle.pos += delta;
    }
}

void DataDescription::rotate(float angle)
{
    auto rotationMatrix = Math::calcRotationMatrix(angle);
    auto center = calcCenter();

    auto rotate = [&](RealVector2D& pos) {
        auto relPos = pos - center;
        auto rotatedRelPos = rotationMatrix * relPos;
        pos = center + rotatedRelPos;
    };
    for (auto& cell : cells) {
        rotate(cell.pos);
    }
    for (auto& particle : particles) {
        rotate(particle.pos);
    }
}

void DataDescription::accelerate(RealVector2D const& velDelta, float angularVelDelta)
{
    auto center = calcCenter();

    auto accelerate = [&](RealVector2D const& pos, RealVector2D& vel) {
        auto relPos = pos - center;
        vel += Physics::tangentialVelocity(relPos, velDelta, angularVelDelta);
    };
    for (auto& cell : cells) {
        accelerate(cell.pos, cell.vel);
    }
    for (auto& particle : particles) {
        accelerate(particle.pos, particle.vel);
    }
}

std::unordered_set<uint64_t> DataDescription::getCellIds() const
{
    std::unordered_set<uint64_t> result;
    for (auto const& cell : cells) {
        result.insert(cell.id);
    }
    return result;
}

DataDescription&
DataDescription::addConnection(uint64_t const& cellId1, uint64_t const& cellId2, std::unordered_map<uint64_t, int>* cache)
{
    auto& cell1 = getCellRef(cellId1, cache);
    auto& cell2 = getCellRef(cellId2, cache);

    auto addConnection = [this, &cache](auto& cell, auto& otherCell) {
        CHECK(cell.connections.size() < cell.maxConnections);

        auto newAngle = Math::angleOfVector(otherCell.pos - cell.pos);

        if (cell.connections.empty()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(otherCell.pos - cell.pos));
            newConnection.angleFromPrevious = 360.0;
            cell.connections.emplace_back(newConnection);
            return;
        }
        if (1 == cell.connections.size()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(otherCell.pos - cell.pos));

            auto connectedCell = getCellRef(cell.connections.front().cellId, cache);
            auto connectedCellDelta = connectedCell.pos - cell.pos;
            auto prevAngle = Math::angleOfVector(connectedCellDelta);
            auto angleDiff = newAngle - prevAngle;
            if (angleDiff >= 0) {
                newConnection.angleFromPrevious = toFloat(angleDiff);
                cell.connections.begin()->angleFromPrevious = 360.0f - toFloat(angleDiff);
            } else {
                newConnection.angleFromPrevious = 360.0f + toFloat(angleDiff);
                cell.connections.begin()->angleFromPrevious = toFloat(-angleDiff);
            }
            cell.connections.emplace_back(newConnection);
            return;
        }

        auto firstConnectedCell = getCellRef(cell.connections.front().cellId, cache);
        auto firstConnectedCellDelta = firstConnectedCell.pos - cell.pos;
        auto angle = Math::angleOfVector(firstConnectedCellDelta);
        auto connectionIt = ++cell.connections.begin();
        while (true) {
            auto nextAngle = angle + connectionIt->angleFromPrevious;

            if ((angle < newAngle && newAngle <= nextAngle) || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
                break;
            }

            ++connectionIt;
            if (connectionIt == cell.connections.end()) {
                connectionIt = cell.connections.begin();
            }
            angle = nextAngle;
            if (angle > 360.0f) {
                angle -= 360.0f;
            }
        }

        ConnectionDescription newConnection;
        newConnection.cellId = otherCell.id;
        newConnection.distance = toFloat(Math::length(otherCell.pos - cell.pos));

        auto angleDiff1 = newAngle - angle;
        if (angleDiff1 < 0) {
            angleDiff1 += 360.0f;
        }
        auto angleDiff2 = connectionIt->angleFromPrevious;

        auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
        newConnection.angleFromPrevious = toFloat(angleDiff2 * factor);
        connectionIt = cell.connections.insert(connectionIt, newConnection);
        ++connectionIt;
        if (connectionIt == cell.connections.end()) {
            connectionIt = cell.connections.begin();
        }
        connectionIt->angleFromPrevious = toFloat(angleDiff2 * (1 - factor));
    };

    addConnection(cell1, cell2);
    addConnection(cell2, cell1);

    return *this;
}

CellDescription& DataDescription::getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>* cache)
{
    if (cache) {
        auto findResult = cache->find(cellId);
        if (findResult != cache->end()) {
            return cells.at(findResult->second);
        }
    }
    for (int i = 0; i < cells.size(); ++i) {
        auto& cell = cells.at(i);
        if (cell.id == cellId) {
            if (cache) {
                cache->emplace(cellId, i);
            }
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}
