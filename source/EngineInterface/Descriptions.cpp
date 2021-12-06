#include "Descriptions.h"

#include <boost/range/adaptors.hpp>

#include "Base/Math.h"
#include "Base/Physics.h"

#include "ChangeDescriptions.h"


bool TokenDescription::operator==(TokenDescription const& other) const {
    return energy == other.energy && data == other.data;
}

namespace
{
    boost::optional<std::list<ConnectionDescription>> convert(
        boost::optional<std::list<ConnectionChangeDescription>> const& connections)
    {
        if (!connections) {
            return boost::none;
        }
        std::list<ConnectionDescription> result;
        for (auto const& connectionChange : *connections) {
            ConnectionDescription connection;
            connection.cellId = connectionChange.cellId;
            connection.distance = connectionChange.distance;
            connection.angleFromPrevious = connectionChange.angleFromPrevious;
            result.emplace_back(connection);
        }
        return result;
    }

}

CellDescription::CellDescription(CellChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<boost::optional<RealVector2D>>(change.pos);
	energy = static_cast<boost::optional<double>>(change.energy);
	maxConnections = static_cast<boost::optional<int>>(change.maxConnections);
    connections =
        convert(static_cast<boost::optional<std::list<ConnectionChangeDescription>>>(change.connectingCells));
	tokenBlocked = change.tokenBlocked.getOptionalValue();	//static_cast<boost::optional<bool>> doesn't work for some reason
	tokenBranchNumber = static_cast<boost::optional<int>>(change.tokenBranchNumber);
	metadata = static_cast<boost::optional<CellMetadata>>(change.metadata);
	cellFeature = static_cast<boost::optional<CellFeatureDescription>>(change.cellFeatures);
	tokens = static_cast<boost::optional<vector<TokenDescription>>>(change.tokens);
    tokenUsages = static_cast<boost::optional<int>>(change.tokenUsages);
}

CellDescription& CellDescription::addToken(TokenDescription const& value)
{
	if (!tokens) {
		tokens = vector<TokenDescription>();
	}
	tokens->push_back(value);
	return *this;
}

CellDescription& CellDescription::addToken(int index, TokenDescription const& value)
{
	if (!tokens) {
		tokens = vector<TokenDescription>();
	}
	tokens->insert(tokens->begin() + index, value);
	return *this;
}

CellDescription& CellDescription::delToken(int index)
{
	CHECK(tokens);
	tokens->erase(tokens->begin() + index);
	return *this;
}

bool CellDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(
               connections->begin(),
               connections->end(),
               [&id](auto const& connection) { return connection.cellId == id; })
        != connections->end();
}

CellDescription2::CellDescription2(CellChangeDescription const& change)
{
    id = change.id;
    pos = *static_cast<boost::optional<RealVector2D>>(change.pos);
    energy = *static_cast<boost::optional<double>>(change.energy);
    maxConnections = *static_cast<boost::optional<int>>(change.maxConnections);

    std::list<ConnectionDescription> connectionList =
        *convert(static_cast<boost::optional<std::list<ConnectionChangeDescription>>>(change.connectingCells));
    connections = std::vector<ConnectionDescription>(connectionList.begin(), connectionList.end());
    tokenBlocked = *change.tokenBlocked.getOptionalValue();  //static_cast<boost::optional<bool>> doesn't work for some reason
    tokenBranchNumber = *static_cast<boost::optional<int>>(change.tokenBranchNumber);
    metadata = *static_cast<boost::optional<CellMetadata>>(change.metadata);
    cellFeature =
        *static_cast<boost::optional<CellFeatureDescription>>(change.cellFeatures);
    for (auto const& token : *static_cast<boost::optional<vector<TokenDescription>>>(change.tokens)) {
        auto newToken = Converter::convert2(token);
        tokens.emplace_back(newToken);
    }
    tokenUsages = *static_cast<boost::optional<int>>(change.tokenUsages);
}

CellDescription2& CellDescription2::addToken(TokenDescription2 const& value)
{
    tokens.emplace_back(value);
    return *this;
}

CellDescription2& CellDescription2::addToken(int index, TokenDescription2 const& value)
{
    tokens.insert(tokens.begin() + index, value);
    return *this;
}

CellDescription2& CellDescription2::delToken(int index)
{
    tokens.erase(tokens.begin() + index);
    return *this;
}

bool CellDescription2::isConnectedTo(uint64_t id) const
{
    return std::find_if(
               connections.begin(),
               connections.end(),
               [&id](auto const& connection) { return connection.cellId == id; })
        != connections.end();
}

ClusterDescription& ClusterDescription::addConnection(
    uint64_t const& cellId1,
    uint64_t const& cellId2,
    std::unordered_map<uint64_t, int>& cache)
{
    auto& cell1 = getCellRef(cellId1, cache);
    auto& cell2 = getCellRef(cellId2, cache);

	auto addConnection = [this, &cache](auto& cell, auto& otherCell) {
        if (!cell.connections) {
            cell.connections = list<ConnectionDescription>();
        }
        CHECK(cell.connections->size() < *cell.maxConnections);

        auto newAngle = Math::angleOfVector(*otherCell.pos - *cell.pos);
        if (cell.id == 281474976710658 && otherCell.id == 281474976710660 && cell.connections
            && cell.connections->size() == 2) {
            int dummy = 0;
        }
        if (cell.id == 281474976710660 && otherCell.id == 281474976710662 && cell.connections
            && cell.connections->size() == 2) {
            int dummy = 0;
        }

        if (cell.connections->empty()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(*otherCell.pos - *cell.pos));
            newConnection.angleFromPrevious = 360.0;
            cell.connections->emplace_back(newConnection);
            return;
        }
        if (1 == cell.connections->size()) {
            ConnectionDescription newConnection;
            newConnection.cellId = otherCell.id;
            newConnection.distance = toFloat(Math::length(*otherCell.pos - *cell.pos));

            auto connectedCell = getCellRef(cell.connections->front().cellId, cache);
            auto connectedCellDelta = *connectedCell.pos - *cell.pos;
            auto prevAngle = Math::angleOfVector(connectedCellDelta);
            auto angleDiff = newAngle - prevAngle;
            if (angleDiff >= 0) {
                newConnection.angleFromPrevious = toFloat(angleDiff);
                cell.connections->begin()->angleFromPrevious = 360.0f - toFloat(angleDiff);
            } else {
                newConnection.angleFromPrevious = 360.0f + toFloat(angleDiff);
                cell.connections->begin()->angleFromPrevious = toFloat(-angleDiff);
            }
            cell.connections->emplace_back(newConnection);
            return;
        }

        auto firstConnectedCell = getCellRef(cell.connections->front().cellId, cache);
        auto firstConnectedCellDelta = *firstConnectedCell.pos - *cell.pos;
        auto angle = Math::angleOfVector(firstConnectedCellDelta);
        auto connectionIt = ++cell.connections->begin();
        while (true) {
            auto nextAngle = angle + connectionIt->angleFromPrevious;

            if ((angle < newAngle && newAngle <= nextAngle)
                || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
                break;
            }

            ++connectionIt;
            if (connectionIt == cell.connections->end()) {
                connectionIt = cell.connections->begin();
            }
            angle = nextAngle;
            if (angle > 360.0f) {
                angle -= 360.0f;
            }
        }

        ConnectionDescription newConnection;
        newConnection.cellId = otherCell.id;
        newConnection.distance = toFloat(Math::length(*otherCell.pos - *cell.pos));

        auto angleDiff1 = newAngle - angle;
        if (angleDiff1 < 0) {
            angleDiff1 += 360.0f;
        }
        auto angleDiff2 = connectionIt->angleFromPrevious;

        auto factor = (angleDiff2 != 0) ? angleDiff1 / angleDiff2 : 0.5f;
        newConnection.angleFromPrevious = toFloat(angleDiff2 * factor);
        connectionIt = cell.connections->insert(connectionIt, newConnection);
        ++connectionIt;
        if (connectionIt == cell.connections->end()) {
            connectionIt = cell.connections->begin();
        }
        connectionIt->angleFromPrevious = toFloat(angleDiff2 * (1 - factor));
    };

    addConnection(cell1, cell2);
    addConnection(cell2, cell1);

    return *this;
}

RealVector2D ClusterDescription::getClusterPosFromCells() const
{
	RealVector2D result;
	if (cells) {
		for (CellDescription const& cell : *cells) {
			result += *cell.pos;
		}
		result /= cells->size();
	}
	return result;
}

CellDescription& ClusterDescription::getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>& cache)
{
    auto findResult = cache.find(cellId);
    if (findResult != cache.end()) {
        return cells->at(findResult->second);
    }
    for (int i = 0; i < cells->size(); ++i) {
        auto& cell = cells->at(i);
        if (cell.id == cellId) {
            cache.emplace(cellId, i);
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

ClusterDescription2& ClusterDescription2::addConnection(
    uint64_t const& cellId1,
    uint64_t const& cellId2,
    std::unordered_map<uint64_t, int>& cache)
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

            if ((angle < newAngle && newAngle <= nextAngle)
                || (angle < (newAngle + 360.0f) && (newAngle + 360.0f) <= nextAngle)) {
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

RealVector2D ClusterDescription2::getClusterPosFromCells() const
{
    RealVector2D result;
    for (auto const& cell : cells) {
        result += cell.pos;
    }
    result /= cells.size();
    return result;
}

CellDescription2& ClusterDescription2::getCellRef(uint64_t const& cellId, std::unordered_map<uint64_t, int>& cache)
{
    auto findResult = cache.find(cellId);
    if (findResult != cache.end()) {
        return cells.at(findResult->second);
    }
    for (int i = 0; i < cells.size(); ++i) {
        auto& cell = cells.at(i);
        if (cell.id == cellId) {
            cache.emplace(cellId, i);
            return cell;
        }
    }
    THROW_NOT_IMPLEMENTED();
}

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<boost::optional<RealVector2D>>(change.pos);
	vel = static_cast<boost::optional<RealVector2D>>(change.vel);
	energy = static_cast<boost::optional<double>>(change.energy);
	metadata = static_cast<boost::optional<ParticleMetadata>>(change.metadata);
}

ParticleDescription2::ParticleDescription2(ParticleChangeDescription const& change)
{
    id = change.id;
    pos = *static_cast<boost::optional<RealVector2D>>(change.pos);
    vel = *static_cast<boost::optional<RealVector2D>>(change.vel);
    energy = *static_cast<boost::optional<double>>(change.energy);
    metadata = *static_cast<boost::optional<ParticleMetadata>>(change.metadata);
}

RealVector2D DataDescription::calcCenter() const
{
	RealVector2D result;
	int numEntities = 0;
	if (clusters) {
		for (auto const& cluster : *clusters) {
			if (cluster.cells) {
				for (auto const& cell : *cluster.cells) {
					result += *cell.pos;
					++numEntities;
				}
			}
		}
	}
	if (particles) {
		for (auto const& particle : *particles) {
			result += *particle.pos;
			++numEntities;
		}
	}
	result /= numEntities;
	return result;
}

void DataDescription::shift(RealVector2D const & delta)
{
	if (clusters) {
		for (auto & cluster : *clusters) {
			if (cluster.cells) {
				for (auto & cell : *cluster.cells) {
					*cell.pos += delta;
				}
			}
		}
	}
	if (particles) {
		for (auto & particle : *particles) {
			*particle.pos += delta;
		}
	}
}

RealVector2D DataDescription2::calcCenter() const
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

void DataDescription2::shift(RealVector2D const& delta)
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

TokenDescription2 Converter::convert2(TokenDescription const& token)
{
    TokenDescription2 result;
    result.data = *token.data;
    result.energy = *token.energy;
    return result;
}

CellDescription2 Converter::convert2(CellDescription const& cell)
{
    CellDescription2 result;
    result.id = cell.id;

    result.pos = *cell.pos;
    result.vel = *cell.vel;
    result.energy = *cell.energy;
    result.maxConnections = *cell.maxConnections;
    result.connections = std::vector<ConnectionDescription>(cell.connections->begin(), cell.connections->end());
    result.tokenBlocked = *cell.tokenBlocked;
    result.tokenBranchNumber = *cell.tokenBranchNumber;
    result.metadata = *cell.metadata;
    result.cellFeature = *cell.cellFeature;
    for (auto const& token : *cell.tokens) {
        result.tokens.emplace_back(convert2(token));
    }
    result.tokenUsages = *cell.tokenUsages;
    return result;
}

ClusterDescription2 Converter::convert2(ClusterDescription const& cluster)
{
    ClusterDescription2 result;
    result.id = cluster.id;
    for (auto const& cell : *cluster.cells) {
        result.cells.emplace_back(convert2(cell));
    }
    return result;
}

ParticleDescription2 Converter::convert2(ParticleDescription const& particle)
{
    ParticleDescription2 result;
    result.id = particle.id;
    result.pos = *particle.pos;
    result.vel = *particle.vel;
    result.energy = *particle.energy;
    result.metadata = *particle.metadata;
    return result;
}

DataDescription2 Converter::convert2(DataDescription const& data)
{
    DataDescription2 result;
    for (auto const& cluster : *data.clusters) {
        result.clusters.emplace_back(convert2(cluster));
    }
    for (auto const& particle : *data.particles) {
        result.particles.emplace_back(convert2(particle));
    }
    return result;
}
