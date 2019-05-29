#pragma once
#include "Definitions.h"
#include "Metadata.h"

struct CellFeatureDescription
{
	Enums::CellFunction::Type type = Enums::CellFunction::COMPUTER;
	QByteArray volatileData;
	QByteArray constData;

	CellFeatureDescription& setType(Enums::CellFunction::Type value) { type = value; return *this; }
	CellFeatureDescription& setVolatileData(QByteArray const &value) { volatileData = value; return *this; }
	CellFeatureDescription& setConstData(QByteArray const &value) { constData = value; return *this; }
	bool operator==(CellFeatureDescription const& other) const {
		return type == other.type
			&& volatileData == other.volatileData
			&& constData == other.constData;
	}
	bool operator!=(CellFeatureDescription const& other) const { return !operator==(other); }
};


struct MODELBASIC_EXPORT TokenDescription
{
	optional<double> energy;
	optional<QByteArray> data;

	TokenDescription& setEnergy(double value) { energy = value; return *this; }
	TokenDescription& setData(QByteArray const &value) { data = value; return *this; }
	bool operator==(TokenDescription const& other) const;
	bool operator!=(TokenDescription const& other) const { return !operator==(other); }
};

#include <iostream>

struct MODELBASIC_EXPORT CellDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<double> energy;
	optional<int> maxConnections;
	optional<list<uint64_t>> connectingCells;
	optional<bool> tokenBlocked;
	optional<int> tokenBranchNumber;
	optional<CellMetadata> metadata;
	optional<CellFeatureDescription> cellFeature;
	optional<vector<TokenDescription>> tokens;

	CellDescription() = default;
	CellDescription(CellChangeDescription const& change);
	CellDescription& setId(uint64_t value) { id = value; return *this; }
	CellDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	CellDescription& setEnergy(double value) { energy = value; return *this; }
	CellDescription& setMaxConnections(int value) { maxConnections = value; return *this; }
	CellDescription& setConnectingCells(list<uint64_t> const& value) { connectingCells = value; return *this; }
	CellDescription& addConnection(uint64_t value);
	CellDescription& setFlagTokenBlocked(bool value) { tokenBlocked = value; return *this; }
	CellDescription& setTokenBranchNumber(int value) { tokenBranchNumber = value; return *this; }
	CellDescription& setMetadata(CellMetadata const& value) { metadata = value; return *this; }
	CellDescription& setCellFeature(CellFeatureDescription const& value) { cellFeature = value; return *this; }
	CellDescription& setTokens(vector<TokenDescription> const& value) { tokens = value; return *this; }
	CellDescription& addToken(TokenDescription const& value);
	CellDescription& addToken(uint index, TokenDescription const& value);
	CellDescription& delToken(uint index);
	QVector2D getPosRelativeTo(ClusterDescription const& cluster) const;
};

struct MODELBASIC_EXPORT ClusterDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> angle;
	optional<double> angularVel;
	optional<ClusterMetadata> metadata;
	optional<vector<CellDescription>> cells;

	ClusterDescription() = default;
    
    ClusterDescription(ClusterChangeDescription const& change);
    ClusterDescription& setId(uint64_t value) { id = value; return *this; }
	ClusterDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	ClusterDescription& setVel(QVector2D const& value) { vel = value; return *this; }
	ClusterDescription& setAngle(double value) { angle = value; return *this; }
	ClusterDescription& setAngularVel(double value) { angularVel = value; return *this; }
	ClusterDescription& setMetadata(ClusterMetadata const& value) { metadata = value; return *this; }
	ClusterDescription& addCells(list<CellDescription> const& value)
	{
		if (cells) {
			cells->insert(cells->end(), value.begin(), value.end());
		}
		else {
			cells = vector<CellDescription>(value.begin(), value.end());
		}
		return *this;
	}
	ClusterDescription& addCell(CellDescription const& value)
	{
		addCells({ value });
		return *this;
	}

	QVector2D getClusterPosFromCells() const;
};

struct MODELBASIC_EXPORT ParticleDescription
{
	uint64_t id = 0;

	optional<QVector2D> pos;
	optional<QVector2D> vel;
	optional<double> energy;
	optional<ParticleMetadata> metadata;

	ParticleDescription() = default;
	ParticleDescription(ParticleChangeDescription const& change);
	ParticleDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleDescription& setPos(QVector2D const& value) { pos = value; return *this; }
	ParticleDescription& setVel(QVector2D const& value) { vel = value; return *this; }
	ParticleDescription& setEnergy(double value) { energy = value; return *this; }
};

struct MODELBASIC_EXPORT DataDescription
{
	optional<vector<ClusterDescription>> clusters;
	optional<vector<ParticleDescription>> particles;

    DataDescription() = default;
	DataDescription& addClusters(list<ClusterDescription> const& value)
	{
		if (clusters) {
			clusters->insert(clusters->end(), value.begin(), value.end());
		}
		else {
			clusters = vector<ClusterDescription>(value.begin(), value.end());
		}
		return *this;
	}
	DataDescription& addCluster(ClusterDescription const& value)
	{
		addClusters({ value });
		return *this;
	}
	DataDescription& addParticle(ParticleDescription const& value)
	{
		if (!particles) {
			particles = vector<ParticleDescription>();
		}
		particles->emplace_back(value);
		return *this;
	}
	void clear()
	{
		clusters = boost::none;
		particles = boost::none;
	}
	bool isEmpty() const
	{
		if (clusters && !clusters->empty()) {
			return false;
		}
		if (particles && !particles->empty()) {
			return false;
		}
		return true;
	}
	QVector2D calcCenter() const;
	void shift(QVector2D const& delta);
};

struct ResolveDescription
{
	bool resolveIds = true;
	bool resolveCellLinks = true;
};

struct DescriptionNavigator
{
	unordered_set<uint64_t> cellIds;
	unordered_set<uint64_t> particleIds;
	map<uint64_t, uint64_t> clusterIdsByCellIds;
	map<uint64_t, int> clusterIndicesByClusterIds;
	map<uint64_t, int> clusterIndicesByCellIds;
	map<uint64_t, int> cellIndicesByCellIds;
	map<uint64_t, int> particleIndicesByParticleIds;

	void update(DataDescription const& data)
	{
		cellIds.clear();
		particleIds.clear();
		clusterIdsByCellIds.clear();
		clusterIndicesByCellIds.clear();
		clusterIndicesByClusterIds.clear();
		cellIndicesByCellIds.clear();
		particleIndicesByParticleIds.clear();

		int clusterIndex = 0;
		if (data.clusters) {
			for (auto const &cluster : *data.clusters) {
				clusterIndicesByClusterIds.insert_or_assign(cluster.id, clusterIndex);
				int cellIndex = 0;
				if (cluster.cells) {
					for (auto const &cell : *cluster.cells) {
						clusterIdsByCellIds.insert_or_assign(cell.id, cluster.id);
						clusterIndicesByCellIds.insert_or_assign(cell.id, clusterIndex);
						cellIndicesByCellIds.insert_or_assign(cell.id, cellIndex);
						cellIds.insert(cell.id);
						++cellIndex;
					}
				}
				++clusterIndex;
			}
		}

		int particleIndex = 0;
		if (data.particles) {
			for (auto const &particle : *data.particles) {
				particleIndicesByParticleIds.insert_or_assign(particle.id, particleIndex);
				particleIds.insert(particle.id);
				++particleIndex;
			}
		}
	}
};
