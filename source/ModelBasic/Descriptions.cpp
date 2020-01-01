#include <QMatrix4x4>

#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "Settings.h"


bool TokenDescription::operator==(TokenDescription const& other) const {
	return energy == other.energy
		&& data == other.data;
}

CellDescription::CellDescription(CellChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<optional<QVector2D>>(change.pos);
	energy = static_cast<optional<double>>(change.energy);
	maxConnections = static_cast<optional<int>>(change.maxConnections);
	connectingCells = static_cast<optional<list<uint64_t>>>(change.connectingCells);
	tokenBlocked = change.tokenBlocked.getOptionalValue();	//static_cast<optional<bool>> doesn't work for some reason
	tokenBranchNumber = static_cast<optional<int>>(change.tokenBranchNumber);
	metadata = static_cast<optional<CellMetadata>>(change.metadata);
	cellFeature = static_cast<optional<CellFeatureDescription>>(change.cellFeatures);
	tokens = static_cast<optional<vector<TokenDescription>>>(change.tokens);
}

CellDescription& CellDescription::addConnection(uint64_t value)
{
	if (!connectingCells) {
		connectingCells = list<uint64_t>();
	}
	connectingCells->push_back(value);
	return *this;
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
    return std::find(connectingCells->begin(), connectingCells->end(), id) != connectingCells->end();
}

ClusterDescription::ClusterDescription(ClusterChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<optional<QVector2D>>(change.pos);
	vel = static_cast<optional<QVector2D>>(change.vel);
	angle = static_cast<optional<double>>(change.angle);
	angularVel = static_cast<optional<double>>(change.angularVel);
	metadata = static_cast<optional<ClusterMetadata>>(change.metadata);
	for (auto const& cellTracker : change.cells) {
		if (!cellTracker.isDeleted()) {
			if (!cells) {
				cells = vector<CellDescription>();
			}
			cells->emplace_back(CellDescription(cellTracker.getValue()));
		}
	}

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

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = static_cast<optional<QVector2D>>(change.pos);
	vel = static_cast<optional<QVector2D>>(change.vel);
	energy = static_cast<optional<double>>(change.energy);
	metadata = static_cast<optional<ParticleMetadata>>(change.metadata);
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
