#include "Descriptions.h"
#include "ChangeDescriptions.h"

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
	cellFunction = change.cellFunction;
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
			cells.emplace_back(CellDescription(cellTracker.getValue()));
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
