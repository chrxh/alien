#ifndef ENTITIES_DESCRIPTIONS_H
#define ENTITIES_DESCRIPTIONS_H

#include "model/features/Descriptions.h"

struct TokenDescription
{
	Editable<double> energy;
	Editable<QByteArray> data;

	TokenDescription& setEnergy(double value) { energy.init(value); return *this; }
	TokenDescription& setData(QByteArray const &value) { data.init(value); return *this; }
};

struct CellDescription
{
	uint64_t id = 0.0;
	Editable<QVector2D> relPos;
	Editable<double> energy = 0.0;
	Editable<int> maxConnections = 0;
	Editable<bool> tokenBlocked = false;
	Editable<int> tokenAccessNumber = 0;
	Editable<CellMetadata> metadata;
	Editable<CellFunctionDescription> cellFunction;
	vector<EditableVec<TokenDescription>> tokens;

	CellDescription& setEnergy(double value) { energy.init(value); return *this; }
	CellDescription& setMaxConnections(int value) { maxConnections.init(value); return *this; }
	CellDescription& setFlagTokenBlocked(bool value) { tokenBlocked.init(value); return *this; }
	CellDescription& setTokenAccessNumber(int value) { tokenAccessNumber.init(value); return *this; }
	CellDescription& setMetadata(CellMetadata const& value) { metadata.init(value); return *this; }
	CellDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction.init(value); return *this; }
};

struct CellClusterDescription
{
	uint64_t id = 0.0;
	Editable<QVector2D> pos;
	Editable<QVector2D> vel;
	Editable<double> angle = 0.0;
	Editable<double> angularVel = 0.0;
	Editable<CellClusterMetadata> metadata;
	vector<EditableVec<CellDescription>> cells;
	vector<EditableVec<pair<uint64_t, uint64_t>>> cellConnections;

	CellClusterDescription& setId(uint64_t value) { id = value; return *this; }
	CellClusterDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	CellClusterDescription& setVel(QVector2D const& value) { vel.init(value); return *this; }
	CellClusterDescription& setAngle(double value) { angle.init(value); return *this; }
	CellClusterDescription& setAngularVel(double value) { angularVel.init(value); return *this; }
	CellClusterDescription& addCell(CellDescription const& c)
	{
		cells.push_back(EditableVec<CellDescription>(EditableVecState::Added, c));
		return *this;
	}
};

struct EnergyParticleDescription
{
	uint64_t id = 0.0;
	Editable<QVector2D> pos;
	Editable<QVector2D> vel;
	Editable<double> energy = 0.0;
	Editable<EnergyParticleMetadata> metadata;

	EnergyParticleDescription& setPos(QVector2D const& p) { pos.init(p); return *this; }
	EnergyParticleDescription& setVel(QVector2D const& v) { vel.init(v); return *this; }
	EnergyParticleDescription& setEnergy(double e) { energy.init(e); return *this; }
};

struct DataDescription
{
	vector<EditableVec<CellClusterDescription>> clusters;
	vector<EditableVec<EnergyParticleDescription>> particles;

	DataDescription& addCellCluster(CellClusterDescription const& c)
	{
		clusters.push_back(EditableVec<CellClusterDescription>(EditableVecState::Added, c));
		return *this;
	}
	DataDescription& addEnergyParticle(EnergyParticleDescription const& e)
	{
		particles.push_back(EditableVec<EnergyParticleDescription>(EditableVecState::Added, e));
		return *this;
	}
	void clear()
	{
		clusters.clear();
		particles.clear();
	}
};

#endif // ENTITIES_DESCRIPTIONS_H
