#include "Descriptions.h"

#include <boost/range/adaptors.hpp>

#include "GenomeDescriptionService.h"
#include "Base/Math.h"
#include "Base/Physics.h"

ConstructorDescription::ConstructorDescription()
{
    genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription());
}

InjectorDescription::InjectorDescription()
{
    genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription());
}

CellType CellDescription::getCellType() const
{
    if (std::holds_alternative<StructureCellDescription>(cellTypeData)) {
        return CellType_Structure;
    } else if (std::holds_alternative<FreeCellDescription>(cellTypeData)) {
        return CellType_Free;
    } else if (std::holds_alternative<BaseDescription>(cellTypeData)) {
        return CellType_Base;
    } else if (std::holds_alternative<DepotDescription>(cellTypeData)) {
        return CellType_Depot;
    } else if (std::holds_alternative<ConstructorDescription>(cellTypeData)) {
        return CellType_Constructor;
    } else if (std::holds_alternative<SensorDescription>(cellTypeData)) {
        return CellType_Sensor;
    } else if (std::holds_alternative<OscillatorDescription>(cellTypeData)) {
        return CellType_Oscillator;
    } else if (std::holds_alternative<AttackerDescription>(cellTypeData)) {
        return CellType_Attacker;
    } else if (std::holds_alternative<InjectorDescription>(cellTypeData)) {
        return CellType_Injector;
    } else if (std::holds_alternative<MuscleDescription>(cellTypeData)) {
        return CellType_Muscle;
    } else if (std::holds_alternative<DefenderDescription>(cellTypeData)) {
        return CellType_Defender;
    } else if (std::holds_alternative<ReconnectorDescription>(cellTypeData)) {
        return CellType_Reconnector;
    } else if (std::holds_alternative<DetonatorDescription>(cellTypeData)) {
        return CellType_Detonator;
    }
    CHECK(false);
}

bool CellDescription::hasGenome() const
{
    auto cellType = getCellType();
    if (cellType == CellType_Constructor || cellType == CellType_Injector) {
        return true;
    }
    return false;
}

std::vector<uint8_t>& CellDescription::getGenomeRef()
{
    auto cellType = getCellType();
    if (cellType == CellType_Constructor) {
        return std::get<ConstructorDescription>(cellTypeData).genome;
    }
    if (cellType == CellType_Injector) {
        return std::get<InjectorDescription>(cellTypeData).genome;
    }
    THROW_NOT_IMPLEMENTED();
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
    auto& cell2 = getCellRef(cellId2, cache);
    return addConnection(cellId1, cellId2, cell2.pos, cache);
}

DataDescription& DataDescription::addConnection(
    uint64_t const& cellId1,
    uint64_t const& cellId2,
    RealVector2D const& refPosCell2,
    std::unordered_map<uint64_t, int>* cache /*= nullptr*/)
{
    auto& cell1 = getCellRef(cellId1, cache);
    auto& cell2 = getCellRef(cellId2, cache);

    auto addConnection = [this,
                          &cache](CellDescription& cell, CellDescription& otherCell, RealVector2D const& cellRefPos, RealVector2D const& otherCellRefPos) {
        CHECK(cell.connections.size() < MAX_CELL_BONDS);

        auto newAngle = Math::angleOfVector(otherCellRefPos - cellRefPos);

        if (cell.connections.empty()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(otherCellRefPos - cellRefPos));
            newConnection.angleFromPrevious = 360.0;
            cell.connections.emplace_back(newConnection);
            return;
        }
        if (1 == cell.connections.size()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

            auto connectedCell = getCellRef(cell.connections.front().cellId, cache);
            auto connectedCellDelta = connectedCell.pos - cellRefPos;
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
        auto firstConnectedCellDelta = firstConnectedCell.pos - cellRefPos;
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
        newConnection.distance = toFloat(Math::length(otherCellRefPos - cellRefPos));

        auto angleDiff1 = newAngle - angle;
        if (angleDiff1 < 0) {
            angleDiff1 += 360.0f;
        }
        auto angleDiff2 = connectionIt->angleFromPrevious;
        if (connectionIt == cell.connections.begin()) {
            connectionIt = cell.connections.end();  // connection at index 0 should be an invariant
        }

        auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
        newConnection.angleFromPrevious = toFloat(angleDiff2 * factor);
        connectionIt = cell.connections.insert(connectionIt, newConnection);
        ++connectionIt;
        if (connectionIt == cell.connections.end()) {
            connectionIt = cell.connections.begin();
        }
        connectionIt->angleFromPrevious = toFloat(angleDiff2 * (1 - factor));
    };

    addConnection(cell1, cell2, cell1.pos, refPosCell2);
    addConnection(cell2, cell1, refPosCell2, cell1.pos);

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
