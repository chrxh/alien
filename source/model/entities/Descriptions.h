#ifndef ENTITIES_DESCRIPTIONS_H
#define ENTITIES_DESCRIPTIONS_H

#include "model/features/Descriptions.h"

struct TokenDescription
{
	double energy = 0.0;
	QByteArray data;

	TokenDescription& setEnergy(double value) { energy = value; return *this; }
	TokenDescription& setData(QByteArray const &value) { data = value; return *this; }
};

struct CellDescription
{
	uint64_t id = 0.0;

	Tracker<QVector2D> pos;
	Tracker<double> energy;
	Tracker<int> maxConnections;
	Tracker<bool> tokenBlocked;
	Tracker<int> tokenAccessNumber;
	Tracker<CellMetadata> metadata;
	Tracker<CellFunctionDescription> cellFunction;
	Tracker<vector<TokenDescription>> tokens;

	CellDescription& setId(uint64_t value) { id = value; return *this; }
	CellDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	CellDescription& setEnergy(double value) { energy.init(value); return *this; }
	CellDescription& setMaxConnections(int value) { maxConnections.init(value); return *this; }
	CellDescription& setFlagTokenBlocked(bool value) { tokenBlocked.init(value); return *this; }
	CellDescription& setTokenAccessNumber(int value) { tokenAccessNumber.init(value); return *this; }
	CellDescription& setMetadata(CellMetadata const& value) { metadata.init(value); return *this; }
	CellDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction.init(value); return *this; }
};

struct CellClusterDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<QVector2D> vel;
	Tracker<double> angle;
	Tracker<double> angularVel;
	Tracker<CellClusterMetadata> metadata;
	vector<TrackerElement<CellDescription>> cells;
	vector<TrackerElement<pair<uint64_t, uint64_t>>> cellConnections;

	CellClusterDescription& setId(uint64_t value) { id = value; return *this; }
	CellClusterDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	CellClusterDescription& setVel(QVector2D const& value) { vel.init(value); return *this; }
	CellClusterDescription& setAngle(double value) { angle.init(value); return *this; }
	CellClusterDescription& setAngularVel(double value) { angularVel.init(value); return *this; }
	CellClusterDescription& addCell(CellDescription const& value)
	{
		cells.emplace_back(TrackerElement<CellDescription>(value, TrackerElementState::Added));
		return *this;
	}
	CellDescription& test(CellDescription & value)
	{
		cells.emplace_back(TrackerElement<CellDescription>(value, TrackerElementState::Retained));
		return value;
	}
	CellClusterDescription& retainCell(CellDescription const& value)
	{
		cells.emplace_back(TrackerElement<CellDescription>(value, TrackerElementState::Retained));
		return *this;
	}
	CellClusterDescription& retainConnection(pair<uint64_t, uint64_t> const& value)
	{
		cellConnections.emplace_back(TrackerElement<pair<uint64_t, uint64_t>>(value, TrackerElementState::Retained));
		return *this;
	}
};

struct EnergyParticleDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<QVector2D> vel;
	Tracker<double> energy;
	Tracker<EnergyParticleMetadata> metadata;

	EnergyParticleDescription& setId(uint64_t value) { id = value; return *this; }
	EnergyParticleDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	EnergyParticleDescription& setVel(QVector2D const& value) { vel.init(value); return *this; }
	EnergyParticleDescription& setEnergy(double value) { energy.init(value); return *this; }
};

struct DataDescription
{
	vector<TrackerElement<CellClusterDescription>> clusters;
	vector<TrackerElement<EnergyParticleDescription>> particles;

	DataDescription& addCellCluster(CellClusterDescription const& value)
	{
		clusters.emplace_back(TrackerElement<CellClusterDescription>(value, TrackerElementState::Added));
		return *this;
	}
	DataDescription& addEnergyParticle(EnergyParticleDescription const& value)
	{
		particles.emplace_back(TrackerElement<EnergyParticleDescription>(value, TrackerElementState::Added));
		return *this;
	}
	void clear()
	{
		clusters.clear();
		particles.clear();
	}
};

struct ResolveDescription
{
	bool resolveCellLinks = false;
};

#endif // ENTITIES_DESCRIPTIONS_H
