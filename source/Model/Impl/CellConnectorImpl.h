#pragma once

#include "Model/Api/CellConnector.h"

class CellConnectorImpl
	: public CellConnector
{
	Q_OBJECT
public:
	CellConnectorImpl(QObject *parent = nullptr) : CellConnector(parent) { }
	virtual ~CellConnectorImpl() = default;

	virtual void init(SpaceMetric *metric, SimulationParameters *parameters, NumberGenerator *numberGen);

	virtual void reconnect(DataDescription &data, list<uint64_t> const &changedCellIds) override;

private:
	list<uint64_t> filterPresentCellIds(list<uint64_t> const& cellIds) const;
	void updateInternals(DataDescription const &data);
	void updateConnectingCells(DataDescription &data, list<uint64_t> const &changedCellIds);
	void reclustering(DataDescription &data, list<uint64_t> const &changedCellIds);

	CellDescription& getCellDescRef(DataDescription &data, uint64_t cellId);
	void removeConnections(DataDescription &data, CellDescription &cellDesc);
	void establishNewConnectionsWithNeighborCells(DataDescription &data, CellDescription &cellDesc);
	void establishNewConnection(CellDescription &cell1, CellDescription &cell2) const;
	double getDistance(CellDescription &cell1, CellDescription &cell2) const;

	list<uint64_t> getCellIdsAtPos(IntVector2D const &pos);

	unordered_set<int> reclusteringSingleClusterAndReturnDiscardedClusterIndices(DataDescription &data
		, int clusterIndex, vector<ClusterDescription> &newClusters);
	void lookUpCell(DataDescription &data, uint64_t cellId, ClusterDescription &newCluster
		, unordered_set<uint64_t> &lookedUpCellIds, unordered_set<uint64_t> &remainingCellIds);

	void setClusterAttributes(DataDescription const& data, ClusterDescription& cluster);
	double calcAngle(DataDescription const& data, ClusterDescription const &changedCellIds) const;

	SpaceMetric *_metric = nullptr;
	SimulationParameters *_parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;

	DescriptionNavigationMaps _navi;
	unordered_map<int, unordered_map<int, list<uint64_t>>> _cellMap;
};
