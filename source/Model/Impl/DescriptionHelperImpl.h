#pragma once

#include "Model/Api/DescriptionHelper.h"

class DescriptionHelperImpl
	: public DescriptionHelper
{
	Q_OBJECT
public:
	DescriptionHelperImpl(QObject *parent = nullptr) : DescriptionHelper(parent) { }
	virtual ~DescriptionHelperImpl() = default;

	virtual void init(SpaceMetric *metric, SimulationParameters *parameters, NumberGenerator *numberGen);

	virtual void reconnect(DataDescription& data, unordered_set<uint64_t> const& changedCellIds) override;
	virtual void recluster(DataDescription& data, unordered_set<uint64_t> const& changedClusterIds) override;
	virtual void makeValid(ClusterDescription& cluster) override;
	virtual void makeValid(ParticleDescription& particle) override;

private:
	list<uint64_t> filterPresentCellIds(unordered_set<uint64_t> const& cellIds) const;
	void updateInternals();
	void updateConnectingCells(list<uint64_t> const &changedCellIds);
	void reclustering(unordered_set<uint64_t> const& clusterIds);

	CellDescription& getCellDescRef(uint64_t cellId);
	void removeConnections(CellDescription &cellDesc);
	void establishNewConnectionsWithNeighborCells(CellDescription &cellDesc);
	void establishNewConnection(CellDescription &cell1, CellDescription &cell2) const;
	double getDistance(CellDescription &cell1, CellDescription &cell2) const;

	list<uint64_t> getCellIdsAtPos(IntVector2D const &pos);

	unordered_set<int> reclusteringSingleClusterAndReturnDiscardedClusterIndices(int clusterIndex, vector<ClusterDescription> &newClusters);
	void lookUpCell(uint64_t cellId, ClusterDescription &newCluster, unordered_set<uint64_t> &lookedUpCellIds, unordered_set<uint64_t> &remainingCellIds);

	void setClusterAttributes(ClusterDescription& cluster);
	double calcAngleBasedOnOldClusters(vector<CellDescription> const & cells) const;
	struct ClusterVelocities {
		QVector2D linearVel;
		double angularVel = 0.0;
	};
	ClusterVelocities calcVelocitiesBasedOnOldClusters(vector<CellDescription> const & cells) const;
	optional<ClusterMetadata> calcMetadataBasedOnOldClusters(vector<CellDescription> const & cells) const;

	SpaceMetric* _metric = nullptr;
	SimulationParameters* _parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;

	DataDescription* _data = nullptr;
	DescriptionNavigator _navi;
	unordered_map<int, unordered_map<int, list<uint64_t>>> _cellMap;
};
