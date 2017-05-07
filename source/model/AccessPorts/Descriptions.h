#ifndef DESCRIPTIONS_H
#define DESCRIPTIONS_H

#include "model/Definitions.h"
#include "model/features/CellFeatureEnums.h"

struct CellFunctionDescription
{
	Enums::CellFunction::Type type;
	QByteArray data;
};

struct TokenDescription
{
	double energy;
	QByteArray data;
};

struct CellDescription
{
	uint64_t id = 0.0;
	QVector2D relPos;
	double energy = 0.0;
	int maxConnections = 0;
	bool allowToken = true;
	int tokenAccessNumber = 0;
	CellMetadata metadata;
	CellFunctionDescription cellFunction;
	vector<TokenDescription> tokens;

	CellDescription& setEnergy(double e) { energy = e; return *this; }
	CellDescription& setMaxConnections(int c) { maxConnections = c; return *this; }
};

struct CellClusterDescription
{
	uint64_t id = 0.0;
	QVector2D pos;
	QVector2D vel;
	double angle = 0.0;
	double angularVel = 0.0;
	CellClusterMetadata metadata;
	vector<CellDescription> cells;
	vector<pair<uint64_t, uint64_t>> cellConnections;

	CellClusterDescription& setPos(QVector2D const& p) { pos = p; return *this; }
	CellClusterDescription& setVel(QVector2D const& v) { vel = v; return *this; }
	CellClusterDescription& addCell(CellDescription const& c) { cells.push_back(c); return *this; }
};

struct EnergyParticleDescription
{
	uint64_t id = 0.0;
	QVector2D pos;
	QVector2D vel;
	double energy = 0.0;
	EnergyParticleMetadata metadata;
};

struct DataDescription
{
	vector<CellClusterDescription> clusters;
	vector<EnergyParticleDescription> particles;

	DataDescription& addCellCluster(CellClusterDescription const& c) { clusters.push_back(c); return *this; }
	DataDescription& addEnergyParticle(EnergyParticleDescription const& e) { particles.push_back(e); return *this; }
};

#endif // DESCRIPTIONS_H
