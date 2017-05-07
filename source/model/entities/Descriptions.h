#ifndef ENTITIES_DESCRIPTIONS_H
#define ENTITIES_DESCRIPTIONS_H

#include "model/features/Descriptions.h"

struct TokenDescription
{
	double energy;
	QByteArray data;

	TokenDescription& setEnergy(double value) { energy = value; return *this; }
	TokenDescription& setData(QByteArray const &value) { data = value; return *this; }
};

struct CellDescription
{
	uint64_t id = 0.0;
	QVector2D relPos;
	double energy = 0.0;
	int maxConnections = 0;
	bool tokenBlocked = false;
	int tokenAccessNumber = 0;
	CellMetadata metadata;
	CellFunctionDescription cellFunction;
	vector<TokenDescription> tokens;

	CellDescription& setEnergy(double value) { energy = value; return *this; }
	CellDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
	CellDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellDescription& setTokenAccessNumber(int value) { tokenAccessNumber = value; return *this; }
	CellDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction = value; return *this; }
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

	EnergyParticleDescription& setPos(QVector2D const& p) { pos = p; return *this; }
	EnergyParticleDescription& setVel(QVector2D const& v) { vel = v; return *this; }
	EnergyParticleDescription& setEnergy(double e) { energy = e; return *this; }
};

struct DataDescription
{
	vector<CellClusterDescription> clusters;
	vector<EnergyParticleDescription> particles;

	DataDescription& addCellCluster(CellClusterDescription const& c) { clusters.push_back(c); return *this; }
	DataDescription& addEnergyParticle(EnergyParticleDescription const& e) { particles.push_back(e); return *this; }
};

#endif // ENTITIES_DESCRIPTIONS_H
