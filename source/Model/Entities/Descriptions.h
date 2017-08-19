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

struct CellDescription
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

	CellDescription& setId(uint64_t value) { id = value; return *this; }
	CellDescription& setPos(QVector2D const& value) { pos.init(value); return *this; }
	CellDescription& setEnergy(double value) { energy.init(value); return *this; }
	CellDescription& setMaxConnections(int value) { maxConnections.init(value); return *this; }
	CellDescription& setConnectingCells(list<uint64_t> const& value) { connectingCells.init(value); return *this; }
	CellDescription& setFlagTokenBlocked(bool value) { tokenBlocked.init(value); return *this; }
	CellDescription& setTokenAccessNumber(int value) { tokenBranchNumber.init(value); return *this; }
	CellDescription& setMetadata(CellMetadata const& value) { metadata.init(value); return *this; }
	CellDescription& setCellFunction(CellFunctionDescription const& value) { cellFunction.init(value); return *this; }
	CellDescription& setAsUnmodified()
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

struct CellClusterDescription
{
	uint64_t id = 0;

	Tracker<QVector2D> pos;
	Tracker<QVector2D> vel;
	Tracker<double> angle;
	Tracker<double> angularVel;
	Tracker<CellClusterMetadata> metadata;
	vector<TrackerElement<CellDescription>> cells;

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
	CellClusterDescription& addCells(list<CellDescription> const& value)
	{
		for (auto const &cell : value) {
			addCell(cell);
		}
		return *this;
	}
	CellClusterDescription& retainCell(CellDescription const& value)
	{
		cells.emplace_back(TrackerElement<CellDescription>(value, TrackerElementState::Unmodified));
		return *this;
	}
	CellClusterDescription& retainCells(list<CellDescription> const& value)
	{
		for (auto const &cell : value) {
			retainCell(cell);
		}
		return *this;
	}
	CellClusterDescription& update(CellClusterDescription const& otherCluster)
	{
		id = otherCluster.id;
		pos = otherCluster.pos;
		vel = otherCluster.vel;
		angle = otherCluster.angle;
		angularVel = otherCluster.angularVel;
		metadata = otherCluster.metadata;

		map<uint64_t, TrackerElement<CellDescription>> cellTrackersByIds;
		for (auto const &cellT : cells) {
			cellTrackersByIds.insert_or_assign(cellT->id, cellT);
		}
		
		cells.clear();
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
	EnergyParticleDescription& setAsUnmodified()
	{
		pos.setAsUnmodified();
		vel.setAsUnmodified();
		energy.setAsUnmodified();
		metadata.setAsUnmodified();
		return *this;
	}
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
	DataDescription& retainCellCluster(CellClusterDescription const& value)
	{
		clusters.emplace_back(TrackerElement<CellClusterDescription>(value, TrackerElementState::Unmodified));
		return *this;
	}
	DataDescription& retainCellClusters(list<CellClusterDescription> const& value)
	{
		for (auto const &cluster : value) {
			retainCellCluster(cluster);
		}
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

struct DescriptionNavigationMaps
{
	set<uint64_t> cellIds;
	set<uint64_t> particleIds;
	map<uint64_t, uint64_t> clusterIdsByCellIds;
	map<uint64_t, int> clusterIndicesByCellIds;
	map<uint64_t, int> cellIndicesByCellIds;
	map<uint64_t, int> particleIndicesByParticleIds;

	void update(DataDescription const& data)
	{
		cellIds.clear();
		particleIds.clear();
		clusterIdsByCellIds.clear();
		clusterIndicesByCellIds.clear();
		cellIndicesByCellIds.clear();
		particleIndicesByParticleIds.clear();

		int clusterIndex = 0;
		for (auto const &cluster : getUndeletedElements(data.clusters)) {
			int cellIndex = 0;
			for (auto const &cell : getUndeletedElements(cluster.cells)) {
				clusterIdsByCellIds.insert_or_assign(cell.id, cluster.id);
				clusterIndicesByCellIds.insert_or_assign(cell.id, clusterIndex);
				cellIndicesByCellIds.insert_or_assign(cell.id, cellIndex);
				cellIds.insert(cell.id);
				++cellIndex;
			}
			++clusterIndex;
		}

		int particleIndex = 0;
		for (auto const &particle : getUndeletedElements(data.particles)) {
			particleIndicesByParticleIds[particle.id] = particleIndex;
			particleIds.insert(particle.id);
			++particleIndex;
		}
	}
};

#endif // ENTITIES_DESCRIPTIONS_H
