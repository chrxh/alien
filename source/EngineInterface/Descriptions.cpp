#include <QMatrix4x4>

#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "Physics.h"


bool TokenDescription::operator==(TokenDescription const& other) const {
	return energy == other.energy
		&& data == other.data;
}

namespace
{
    boost::optional<list<ConnectionDescription>> convert(
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
	pos = static_cast<boost::optional<QVector2D>>(change.pos);
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

CellDescription& CellDescription::addToken(uint index, TokenDescription const& value)
{
	if (!tokens) {
		tokens = vector<TokenDescription>();
	}
	tokens->insert(tokens->begin() + index, value);
	return *this;
}

CellDescription& CellDescription::delToken(uint index)
{
	CHECK(tokens);
	tokens->erase(tokens->begin() + index);
	return *this;
}

QVector2D CellDescription::getPosRelativeTo(ClusterDescription const & cluster) const
{
	QMatrix4x4 transform;
	transform.setToIdentity();
	transform.translate(*cluster.pos);
	transform.rotate(*cluster.angle, 0.0, 0.0, 1.0);
	transform = transform.inverted();
	return transform.map(QVector3D(*pos)).toVector2D();
}

bool CellDescription::isConnectedTo(uint64_t id) const
{
    return std::find_if(
               connections->begin(),
               connections->end(),
               [&id](auto const& connection) { return connection.cellId == id; })
        != connections->end();
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

        list<ConnectionDescription> newConnections;
        auto newAngle = Physics::angleOfVector(*otherCell.pos - *cell.pos);

        float sumPrevAngle = 0;
        float sumAngle = 0;
        auto connectionIt = cell.connections->begin();
        for (int i = 0; i <= cell.connections->size(); ++i) {
            if (i < cell.connections->size()) {
                auto connectedCell = getCellRef(connectionIt->cellId, cache);
                sumAngle += Physics::angleOfVector(*connectedCell.pos - *cell.pos);
            } else {
                sumAngle = 361;
            }

            if (newAngle < sumAngle) {
                ConnectionDescription newConnection;
                newConnection.cellId = cell.id;
                newConnection.distance = (*otherCell.pos - *cell.pos).length();
                newConnection.angleFromPrevious = sumPrevAngle - newAngle;
                newConnections.emplace_back(newConnection);

                bool first = true;
                for (; connectionIt != cell.connections->end(); ++connectionIt) {

                    auto connection = *connectionIt;
                    if (first) {
                        connection.angleFromPrevious = connection.angleFromPrevious - newConnection.angleFromPrevious;
                        first = false;
                    }
                    newConnections.emplace_back(connection);
                }
                break;
            }

            newConnections.emplace_back(*connectionIt);
            ++connectionIt;
            sumPrevAngle = sumAngle;
        }
        cell.connections = newConnections;
    };

    addConnection(cell1, cell2);
    addConnection(cell2, cell1);

    return *this;
}

QVector2D ClusterDescription::getClusterPosFromCells() const
{
	QVector2D result;
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

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<boost::optional<QVector2D>>(change.pos);
	vel = static_cast<boost::optional<QVector2D>>(change.vel);
	energy = static_cast<boost::optional<double>>(change.energy);
	metadata = static_cast<boost::optional<ParticleMetadata>>(change.metadata);
}

QVector2D DataDescription::calcCenter() const
{
	QVector2D result;
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

void DataDescription::shift(QVector2D const & delta)
{
	if (clusters) {
		for (auto & cluster : *clusters) {
			*cluster.pos += delta;
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
