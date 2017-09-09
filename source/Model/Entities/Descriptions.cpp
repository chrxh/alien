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

bool CellDescription::operator==(CellChangeDescription const & other) const
{
	return pos == other.pos && energy == other.energy && maxConnections == other.maxConnections
		&& connectingCells == other.connectingCells && tokenBlocked == other.tokenBlocked
		&& tokenBranchNumber == other.tokenBranchNumber && metadata == other.metadata
		&& cellFunction == other.cellFunction && tokens == other.tokens;
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

bool ClusterDescription::operator==(ClusterDescription const & other) const
{
	return pos == other.pos && vel == other.vel && angle == other.angle && angularVel == other.angularVel
		&& metadata == other.metadata && cells == other.cells;
}

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = change.pos;
	vel = change.vel;
	energy = change.energy;
	metadata = change.metadata;
}

bool ParticleDescription::operator==(ParticleDescription const & other) const
{
	return pos == other.pos && vel == other.vel && energy == other.energy && metadata == other.metadata;
}

bool DataDescription::operator==(DataDescription const & other) const
{
	return clusters == other.clusters && particles == other.particles;
}
