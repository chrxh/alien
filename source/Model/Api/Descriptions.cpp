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
	tokenBlocked = static_cast<optional<bool>>(change.tokenBlocked);
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
