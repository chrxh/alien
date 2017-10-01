#pragma once

#include "Descriptions.h"

struct MODEL_EXPORT CellChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<double> energy;
	optional<int> maxConnections;
	optional<list<uint64_t>> connectingCells;
	optional<bool> tokenBlocked;
	optional<int> tokenBranchNumber;
	optional<CellMetadata> metadata;
	optional<CellFeatureDescription> cellFunction;
	optional<vector<TokenDescription>> tokens;

	CellChangeDescription() = default;
	CellChangeDescription(CellDescription const& desc);
	CellChangeDescription(CellDescription const& before, CellDescription const& after);

	bool isEmpty() const;
	CellChangeDescription& setId(uint64_t value) { id = value; return *this; }
	CellChangeDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	CellChangeDescription& setEnergy(double value) { energy = value; return *this; }
	CellChangeDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
	CellChangeDescription& setConnectingCells(list<uint64_t> const& value) { connectingCells = value; return *this; }
	CellChangeDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellChangeDescription& setTokenBranchNumber(int value) { tokenBranchNumber = value; return *this; }
	CellChangeDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellChangeDescription& setCellFunction(CellFeatureDescription const& value) { cellFunction = value; return *this; }
};

struct MODEL_EXPORT ClusterChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> angle;
	optional<double> angularVel;
	optional<ClusterMetadata> metadata;
	vector<ChangeTracker<CellChangeDescription>> cells;

	ClusterChangeDescription() = default;
	ClusterChangeDescription(ClusterDescription const& desc);
	ClusterChangeDescription(ClusterDescription const& before, ClusterDescription const& after);

	bool isEmpty() const;
	ClusterChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ClusterChangeDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	ClusterChangeDescription& setVel(QVector2D const& value) { vel = value; return *this; }
	ClusterChangeDescription& setAngle(double value) { angle = value; return *this; }
	ClusterChangeDescription& setAngularVel(double value) { angularVel = value; return *this; }
	ClusterChangeDescription& addNewCell(CellChangeDescription const& value)
	{
		cells.emplace_back(ChangeTracker<CellChangeDescription>(value, ChangeTracker<CellChangeDescription>::State::Added));
		return *this;
	}
	ClusterChangeDescription& addNewCells(list<CellChangeDescription> const& value)
	{
		for (auto const &cell : value) {
			addNewCell(cell);
		}
		return *this;
	}
	ClusterChangeDescription& addModifiedCell(CellChangeDescription const& value)
	{
		cells.emplace_back(ChangeTracker<CellChangeDescription>(value, ChangeTracker<CellChangeDescription>::State::Modified));
		return *this;
	}
	ClusterChangeDescription& addModifiedCells(list<CellChangeDescription> const& value)
	{
		for (auto const &cell : value) {
			addModifiedCell(cell);
		}
		return *this;
	}
	ClusterChangeDescription& addDeletedCell(CellChangeDescription const& value)
	{
		cells.emplace_back(ChangeTracker<CellChangeDescription>(value, ChangeTracker<CellChangeDescription>::State::Deleted));
		return *this;
	}
	QVector2D getPosBefore() const;

private:
	optional<QVector2D> _posBefore;
};

struct MODEL_EXPORT ParticleChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> energy;
	optional<ParticleMetadata> metadata;

	ParticleChangeDescription() = default;
	ParticleChangeDescription(ParticleDescription const& desc);
	ParticleChangeDescription(ParticleDescription const& before, ParticleDescription const& after);

	bool isEmpty() const;
	QVector2D getPosBefore() const;
	ParticleChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleChangeDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	ParticleChangeDescription& setVel(QVector2D const& value) { vel = value; return *this; }
	ParticleChangeDescription& setEnergy(double value) { energy = value; return *this; }

private:
	optional<QVector2D> _posBefore;
};

struct MODEL_EXPORT DataChangeDescription
{
	vector<ChangeTracker<ClusterChangeDescription>> clusters;
	vector<ChangeTracker<ParticleChangeDescription>> particles;

	DataChangeDescription() = default;
	DataChangeDescription(DataDescription const& desc);
	DataChangeDescription(DataDescription const& dataBefore, DataDescription const& dataAfter);

	DataChangeDescription& addNewCluster(ClusterChangeDescription const& value)
	{
		clusters.emplace_back(ChangeTracker<ClusterChangeDescription>(value, ChangeTracker<ClusterChangeDescription>::State::Added));
		return *this;
	}
	DataChangeDescription& addModifiedCluster(ClusterChangeDescription const& value)
	{
		clusters.emplace_back(ChangeTracker<ClusterChangeDescription>(value, ChangeTracker<ClusterChangeDescription>::State::Modified));
		return *this;
	}
	DataChangeDescription& addModifiedClusters(list<ClusterChangeDescription> const& value)
	{
		for (auto const &cluster : value) {
			addModifiedCluster(cluster);
		}
		return *this;
	}
	DataChangeDescription& addDeletedCluster(ClusterChangeDescription const& value)
	{
		clusters.emplace_back(ChangeTracker<ClusterChangeDescription>(value, ChangeTracker<ClusterChangeDescription>::State::Deleted));
		return *this;
	}
	DataChangeDescription& addNewParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(ChangeTracker<ParticleChangeDescription>(value, ChangeTracker<ParticleChangeDescription>::State::Added));
		return *this;
	}
	DataChangeDescription& addModifiedParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(ChangeTracker<ParticleChangeDescription>(value, ChangeTracker<ParticleChangeDescription>::State::Modified));
		return *this;
	}
	DataChangeDescription& addDeletedParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(ChangeTracker<ParticleChangeDescription>(value, ChangeTracker<ParticleChangeDescription>::State::Deleted));
		return *this;
	}
	void clear()
	{
		clusters.clear();
		particles.clear();
	}
};


