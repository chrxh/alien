#pragma once

#include "Model/CellConnector.h"

class CellConnectorImpl
	: public CellConnector
{
	Q_OBJECT
public:
	CellConnectorImpl(QObject *parent = nullptr) : CellConnector(parent) { }
	virtual ~CellConnectorImpl() = default;

	virtual void init(SpaceMetricApi *metric, SimulationParameters *parameters, NumberGenerator *numberGen);

	virtual void reconnect(DataChangeDescription &data) override;

private:
	void updateInternals(DataChangeDescription const &data);
	void updateConnectingCells(DataChangeDescription &data);
	void reclustering(DataChangeDescription &data);

	CellChangeDescription& getCellDescRef(DataChangeDescription &data, uint64_t cellId);
	void removeConnections(DataChangeDescription &data, CellChangeDescription &cellDesc);
	void establishNewConnectionsWithNeighborCells(DataChangeDescription &data, CellChangeDescription &cellDesc);
	void establishNewConnection(CellChangeDescription &cell1, CellChangeDescription &cell2);
	double getDistance(CellChangeDescription &cell1, CellChangeDescription &cell2);

	list<uint64_t> getCellIdsAtPos(IntVector2D const &pos);

	unordered_set<int> reclusteringSingleClusterAndReturnModifiedClusterIndices(DataChangeDescription &data, int clusterIndex);
	void lookUpCell(DataChangeDescription &data, uint64_t cellId, ClusterChangeDescription &newCluster
		, unordered_set<uint64_t> &lookedUpCellIds, unordered_set<uint64_t> &remainingCellIds);

	SpaceMetricApi *_metric = nullptr;
	SimulationParameters *_parameters = nullptr;
	NumberGenerator* _numberGen = nullptr;

	DescriptionNavigationMaps _navi;
	unordered_map<int, unordered_map<int, list<uint64_t>>> _cellMap;
};
