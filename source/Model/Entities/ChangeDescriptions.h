#pragma once

#include "Model/Features/Descriptions.h"

#include "Descriptions.h"

struct CellChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<double> energy;
	optional<int> maxConnections;
	optional<list<uint64_t>> connectingCells;
	optional<bool> tokenBlocked;
	optional<int> tokenBranchNumber;
	optional<CellMetadata> metadata;
	optional<CellFunctionDescription> cellFunction;
	optional<vector<TokenDescription>> tokens;

	CellChangeDescription& setId(uint64_t value) { id = value; return *this; }
	CellChangeDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	CellChangeDescription& setEnergy(double value) { energy = value; return *this; }
	CellChangeDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
	CellChangeDescription& setConnectingCells(list<uint64_t> const& value) { connectingCells = value; return *this; }
	CellChangeDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellChangeDescription& setTokenBranchNumber(int value) { tokenBranchNumber = value; return *this; }
	CellChangeDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellChangeDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction = value; return *this; }
};

struct ClusterChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> angle;
	optional<double> angularVel;
	optional<CellClusterMetadata> metadata;
	vector<ChangeTracker<CellChangeDescription>> cells;

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
};

struct ParticleChangeDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> energy;
	optional<EnergyParticleMetadata> metadata;

	ParticleChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleChangeDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	ParticleChangeDescription& setVel(QVector2D const& value) { vel = value; return *this; }
	ParticleChangeDescription& setEnergy(double value) { energy = value; return *this; }
};

struct DataChangeDescription
{
	vector<ChangeTracker<ClusterChangeDescription>> clusters;
	vector<ChangeTracker<ParticleChangeDescription>> particles;

	DataChangeDescription() = default;
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
	DataChangeDescription& addDeletedCluster(uint64_t id)
	{
		ClusterChangeDescription cluster;
		cluster.id = id;
		clusters.emplace_back(ChangeTracker<ClusterChangeDescription>(cluster, ChangeTracker<ClusterChangeDescription>::State::Deleted));
		return *this;
	}
	DataChangeDescription& addParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(ChangeTracker<ParticleChangeDescription>(value, ChangeTracker<ParticleChangeDescription>::State::Added));
		return *this;
	}
	void clear()
	{
		clusters.clear();
		particles.clear();
	}
};


