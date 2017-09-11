#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "Model/Settings.h"

namespace
{
	template<typename T>
	bool isCompatible(T const& a, T const& b)
	{
		return a == b;
	}

	template<>
	bool isCompatible<QVector2D>(QVector2D const& vec1, QVector2D const& vec2)
	{
		return std::abs(vec1.x() - vec2.x()) < ALIEN_PRECISION
			&& std::abs(vec1.y() - vec2.y()) < ALIEN_PRECISION;
	}

	template<>
	bool isCompatible<double>(double const& a, double const& b)
	{
		return std::abs(a - b) < ALIEN_PRECISION;
	}

	template<>
	bool isCompatible<CellDescription>(CellDescription const& a, CellDescription const& b)
	{
		return a.isCompatibleWith(b);
	}

	template<>
	bool isCompatible<ClusterDescription>(ClusterDescription const& a, ClusterDescription const& b)
	{
		return a.isCompatibleWith(b);
	}

	template<>
	bool isCompatible<ParticleDescription>(ParticleDescription const& a, ParticleDescription const& b)
	{
		return a.isCompatibleWith(b);
	}

	template<typename T>
	bool isCompatible(optional<T> const& a, optional<T> const& b)
	{
		if (!a || !b) {
			return true;
		}
		return isCompatible(*a, *b);
	}

	template<typename T>
	bool isCompatible(vector<T> const& a, vector<T> const& b)
	{
		if (a.size() != b.size()) {
			false;
		}
		for (int i = 0; i < a.size(); ++i) {
			if (!isCompatible(a.at(i), b.at(i))) {
				return false;
			}
		}
		return true;
	}
}


bool TokenDescription::operator==(TokenDescription const& other) const {
	return energy == other.energy
		&& data == other.data;
}

bool TokenDescription::isCompatibleWith(TokenDescription const & other) const
{
	return isCompatible(energy, other.energy)
		&& isCompatible(data, other.data);
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
	cellFunction = change.cellFunction;
	tokens = change.tokens;
}

bool CellDescription::isCompatibleWith(CellChangeDescription const & other) const
{
	return isCompatible(pos, other.pos)
		&& isCompatible(energy, other.energy)
		&& isCompatible(maxConnections, other.maxConnections)
		&& isCompatible(connectingCells, other.connectingCells)
		&& isCompatible(tokenBlocked, other.tokenBlocked)
		&& isCompatible(tokenBranchNumber, other.tokenBranchNumber)
		&& isCompatible(metadata, other.metadata)
		&& isCompatible(cellFunction, other.cellFunction)
		&& isCompatible(tokens, other.tokens)
		;
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

bool ClusterDescription::isCompatibleWith(ClusterDescription const & other) const
{
	return isCompatible(pos, other.pos)
		&& isCompatible(vel, other.vel)
		&& isCompatible(angle, other.angle) 
		&& isCompatible(angularVel, other.angularVel)
		&& isCompatible(metadata, other.metadata)
		&& isCompatible(cells, other.cells);
}

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = change.pos;
	vel = change.vel;
	energy = change.energy;
	metadata = change.metadata;
}

bool ParticleDescription::isCompatibleWith(ParticleDescription const & other) const
{
	return isCompatible(pos, other.pos)
		&& isCompatible(vel, other.vel)
		&& isCompatible(energy, other.energy)
		&& isCompatible(metadata, other.metadata);
}

bool DataDescription::isCompatibleWith(DataDescription const & other) const
{
	return isCompatible(clusters, other.clusters)
		&& isCompatible(particles, other.particles);
}
