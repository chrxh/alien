#ifndef ENTITIES_DESCRIPTIONS_H
#define ENTITIES_DESCRIPTIONS_H

#include "Model/Features/Descriptions.h"

struct TokenDescription
{
	double energy = 0.0;
	QByteArray data;

	TokenDescription& setEnergy(double value) { energy = value; return *this; }
	TokenDescription& setData(QByteArray const &value) { data = value; return *this; }
};

struct CellChangeDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<double> energy;
	Tracker<int> maxConnections;
	Tracker<list<uint64_t>> connectingCells;
	Tracker<bool> tokenBlocked;
	Tracker<int> tokenBranchNumber;
	Tracker<CellMetadata> metadata;
	Tracker<CellFunctionDescription> cellFunction;
	Tracker<vector<TokenDescription>> tokens;

	CellChangeDescription& setId(uint64_t value) { id = value; return *this; }
	CellChangeDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	CellChangeDescription& setEnergy(double value) { energy.init(value); return *this; }
	CellChangeDescription& setMaxConnections(int value) { maxConnections.init(value); return *this; }
	CellChangeDescription& setConnectingCells(list<uint64_t> const& value) { connectingCells.init(value); return *this; }
	CellChangeDescription& setFlagTokenBlocked(bool value) { tokenBlocked.init(value); return *this; }
	CellChangeDescription& setTokenBranchNumber(int value) { tokenBranchNumber.init(value); return *this; }
	CellChangeDescription& setMetadata(CellMetadata const& value) { metadata.init(value); return *this; }
	CellChangeDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction.init(value); return *this; }
	CellChangeDescription& setAsUnmodified()
	{
		pos.setAsUnmodified();
		energy.setAsUnmodified();
		maxConnections.setAsUnmodified();
		connectingCells.setAsUnmodified();
		tokenBlocked.setAsUnmodified();
		tokenBranchNumber.setAsUnmodified();
		metadata.setAsUnmodified();
		cellFunction.setAsUnmodified();
		tokens.setAsUnmodified();
		return *this;
	}
};

struct ClusterChangeDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<QVector2D> vel;
	Tracker<double> angle;
	Tracker<double> angularVel;
	Tracker<CellClusterMetadata> metadata;
	vector<TrackerElement<CellChangeDescription>> cells;

	ClusterChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ClusterChangeDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	ClusterChangeDescription& setVel(QVector2D const& value) { vel.init(value); return *this; }
	ClusterChangeDescription& setAngle(double value) { angle.init(value); return *this; }
	ClusterChangeDescription& setAngularVel(double value) { angularVel.init(value); return *this; }
	ClusterChangeDescription& addCell(CellChangeDescription const& value)
	{
		cells.emplace_back(TrackerElement<CellChangeDescription>(value, TrackerElementState::Added));
		return *this;
	}
	ClusterChangeDescription& addCells(list<CellChangeDescription> const& value)
	{
		for (auto const &cell : value) {
			addCell(cell);
		}
		return *this;
	}
	ClusterChangeDescription& retainCell(CellChangeDescription const& value)
	{
		cells.emplace_back(TrackerElement<CellChangeDescription>(value, TrackerElementState::Unmodified));
		return *this;
	}
	ClusterChangeDescription& retainCells(list<CellChangeDescription> const& value)
	{
		for (auto const &cell : value) {
			retainCell(cell);
		}
		return *this;
	}
	ClusterChangeDescription& update(ClusterChangeDescription const& otherCluster)
	{
		pos = otherCluster.pos;
		vel = otherCluster.vel;
		angle = otherCluster.angle;
		angularVel = otherCluster.angularVel;
		metadata = otherCluster.metadata;

		map<uint64_t, TrackerElement<CellChangeDescription>> cellTrackersByIds;
		vector<TrackerElement<CellChangeDescription>> deletedCellTrackers;
		for (auto const &cellT : cells) {
			if (!cellT.isDeleted()) {
				cellTrackersByIds.insert_or_assign(cellT->id, cellT);
			}
			else {
				deletedCellTrackers.push_back(cellT);
			}
		}
		
		cells = deletedCellTrackers;
		for (auto cellT : otherCluster.cells) {
			auto cellDescIter = cellTrackersByIds.find(cellT->id);
			if (cellDescIter != cellTrackersByIds.end()) {
				if (cellT.isAdded()) {
					cellT.setAsModified();
				}
				cellTrackersByIds.erase(cellDescIter);
			}
			cells.emplace_back(cellT);
		}
		for (auto const &cellTAndId : cellTrackersByIds) {
			auto cellT = cellTAndId.second;
			cellT.setAsDeleted();
			cells.emplace_back(cellT);
		}
		return *this;
	}
};

struct ParticleChangeDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<QVector2D> vel;
	Tracker<double> energy;
	Tracker<EnergyParticleMetadata> metadata;

	ParticleChangeDescription& setId(uint64_t value) { id = value; return *this; }
	ParticleChangeDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	ParticleChangeDescription& setVel(QVector2D const& value) { vel.init(value); return *this; }
	ParticleChangeDescription& setEnergy(double value) { energy.init(value); return *this; }
	ParticleChangeDescription& setAsUnmodified()
	{
		pos.setAsUnmodified();
		vel.setAsUnmodified();
		energy.setAsUnmodified();
		metadata.setAsUnmodified();
		return *this;
	}
};

struct DataChangeDescription
{
	vector<TrackerElement<ClusterChangeDescription>> clusters;
	vector<TrackerElement<ParticleChangeDescription>> particles;

	DataChangeDescription& addCellCluster(ClusterChangeDescription const& value)
	{
		clusters.emplace_back(TrackerElement<ClusterChangeDescription>(value, TrackerElementState::Added));
		return *this;
	}
	DataChangeDescription& retainCellCluster(ClusterChangeDescription const& value)
	{
		clusters.emplace_back(TrackerElement<ClusterChangeDescription>(value, TrackerElementState::Unmodified));
		return *this;
	}
	DataChangeDescription& retainCellClusters(list<ClusterChangeDescription> const& value)
	{
		for (auto const &cluster : value) {
			retainCellCluster(cluster);
		}
		return *this;
	}
	DataChangeDescription& addEnergyParticle(ParticleChangeDescription const& value)
	{
		particles.emplace_back(TrackerElement<ParticleChangeDescription>(value, TrackerElementState::Added));
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

struct DescriptionNavigationMaps
{
	set<uint64_t> cellIds;
	set<uint64_t> particleIds;
	map<uint64_t, uint64_t> clusterIdsByCellIds;
	map<uint64_t, int> clusterIndicesByCellIds;
	map<uint64_t, int> cellIndicesByCellIds;
	map<uint64_t, int> particleIndicesByParticleIds;

	void update(DataChangeDescription const& data)
	{
		cellIds.clear();
		particleIds.clear();
		clusterIdsByCellIds.clear();
		clusterIndicesByCellIds.clear();
		cellIndicesByCellIds.clear();
		particleIndicesByParticleIds.clear();

		int clusterIndex = 0;
		for (auto const &clusterT : data.clusters) {
			if (!clusterT.isDeleted()) {
				int cellIndex = 0;
				for (auto const &cellT : clusterT->cells) {
					if (!cellT.isDeleted()) {
						clusterIdsByCellIds.insert_or_assign(cellT->id, clusterT->id);
						clusterIndicesByCellIds.insert_or_assign(cellT->id, clusterIndex);
						cellIndicesByCellIds.insert_or_assign(cellT->id, cellIndex);
						cellIds.insert(cellT->id);
					}
					++cellIndex;
				}
			}
			++clusterIndex;
		}

		int particleIndex = 0;
		for (auto const &particleT : data.particles) {
			if (!particleT.isDeleted()) {
				particleIndicesByParticleIds[particleT->id] = particleIndex;
				particleIds.insert(particleT->id);
			}
			++particleIndex;
		}
	}
};

#endif // ENTITIES_DESCRIPTIONS_H
