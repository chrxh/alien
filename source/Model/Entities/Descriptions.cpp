#include "Descriptions.h"
#include "ChangeDescriptions.h"
#include "Model/Settings.h"

namespace
{
	template<typename T>
	bool isCompatibleFunc(T const& a, T const& b)
	{
		return a == b;
	}

	template<>
	bool isCompatibleFunc<QVector2D>(QVector2D const& vec1, QVector2D const& vec2)
	{
		return std::abs(vec1.x() - vec2.x()) < ALIEN_PRECISION
			&& std::abs(vec1.y() - vec2.y()) < ALIEN_PRECISION;
	}

	template<>
	bool isCompatibleFunc<double>(double const& a, double const& b)
	{
		return std::abs(a - b) < ALIEN_PRECISION;
	}

	template<>
	bool isCompatibleFunc<CellDescription>(CellDescription const& a, CellDescription const& b)
	{
		return a.isCompatible(b);
	}

	template<>
	bool isCompatibleFunc<ClusterDescription>(ClusterDescription const& a, ClusterDescription const& b)
	{
		return a.isCompatible(b);
	}

	template<>
	bool isCompatibleFunc<ParticleDescription>(ParticleDescription const& a, ParticleDescription const& b)
	{
		return a.isCompatible(b);
	}

	template<typename T>
	bool isCompatibleFunc(optional<T> const& a, optional<T> const& b)
	{
		if (!a || !b) {
			return true;
		}
		return isCompatibleFunc(*a, *b);
	}

	template<typename T>
	bool isCompatibleFunc(vector<T> const& a, vector<T> const& b)
	{
		if (a.size() != b.size()) {
			false;
		}
		for (int i = 0; i < a.size(); ++i) {
			if (!isCompatibleFunc(a.at(i), b.at(i))) {
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

bool TokenDescription::isCompatible(TokenDescription const & other) const
{
	return isCompatibleFunc(energy, other.energy)
		&& isCompatibleFunc(data, other.data);
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

bool CellDescription::isCompatible(CellChangeDescription const & other) const
{
	return isCompatibleFunc(pos, other.pos)
		&& isCompatibleFunc(energy, other.energy)
		&& isCompatibleFunc(maxConnections, other.maxConnections)
		&& isCompatibleFunc(connectingCells, other.connectingCells)
		&& isCompatibleFunc(tokenBlocked, other.tokenBlocked)
		&& isCompatibleFunc(tokenBranchNumber, other.tokenBranchNumber)
		&& isCompatibleFunc(metadata, other.metadata)
		&& isCompatibleFunc(cellFunction, other.cellFunction)
		&& isCompatibleFunc(tokens, other.tokens)
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

bool ClusterDescription::isCompatible(ClusterDescription const & other) const
{
	return isCompatibleFunc(pos, other.pos)
		&& isCompatibleFunc(vel, other.vel)
		&& isCompatibleFunc(angle, other.angle) 
		&& isCompatibleFunc(angularVel, other.angularVel)
		&& isCompatibleFunc(metadata, other.metadata)
		&& isCompatibleFunc(cells, other.cells);
}

ParticleDescription::ParticleDescription(ParticleChangeDescription const & change)
{
	id = change.id;
	pos = change.pos;
	vel = change.vel;
	energy = change.energy;
	metadata = change.metadata;
}

bool ParticleDescription::isCompatible(ParticleDescription const & other) const
{
	return isCompatibleFunc(pos, other.pos)
		&& isCompatibleFunc(vel, other.vel)
		&& isCompatibleFunc(energy, other.energy)
		&& isCompatibleFunc(metadata, other.metadata);
}

bool DataDescription::isCompatible(DataDescription const & other) const
{
	return isCompatibleFunc(clusters, other.clusters)
		&& isCompatibleFunc(particles, other.particles);
}
