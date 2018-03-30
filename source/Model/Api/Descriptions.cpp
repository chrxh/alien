#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "Model/Api/Settings.h"


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
	tokenBlocked = change.tokenBlocked.getOptionalValue();	//static_cast<optional<bool>> doesn't work for same reason
	tokenBranchNumber = static_cast<optional<int>>(change.tokenBranchNumber);
	metadata = static_cast<optional<CellMetadata>>(change.metadata);
	cellFeature = static_cast<optional<CellFeatureDescription>>(change.cellFeatures);
	tokens = static_cast<optional<vector<TokenDescription>>>(change.tokens);
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
