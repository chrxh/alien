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
	pos = change.pos;
	energy = change.energy;
	maxConnections = change.maxConnections;
	connectingCells = change.connectingCells;
	tokenBlocked = change.tokenBlocked;
	tokenBranchNumber = change.tokenBranchNumber;
	metadata = change.metadata;
	cellFeature = change.cellFunction;
	tokens = change.tokens;
}

ClusterDescription::ClusterDescription(ClusterChangeDescription const & change)
{
	id = change.id;
	pos = change.pos;
	vel = change.vel;
	angle = change.angle;
	angularVel = change.angularVel;
	metadata = change.metadata;
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
	pos = change.pos;
	vel = change.vel;
	energy = change.energy;
	metadata = change.metadata;
}
