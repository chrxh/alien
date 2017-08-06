#pragma once

#include "Model/CellConnector.h"

class CellConnectorImpl
	: public CellConnector
{
	Q_OBJECT
public:
	CellConnectorImpl(QObject *parent = nullptr) : CellConnector(parent) { }
	virtual ~CellConnectorImpl() = default;

	virtual void init(SpaceMetricApi* metric, SimulationParameters *parameters);

	virtual void reconnect(DataDescription &data) override;

private:
	void updateInternals(DataDescription const &data);
	void updateConnectingCells(DataDescription &data);
	void reclustering(DataDescription &data);

	CellDescription& getCellDescRef(DataDescription &data, uint64_t cellId);
	void removeConnections(DataDescription &data, CellDescription &cellDesc);
	void establishNewConnectionsWithNeighborCells(DataDescription &data, CellDescription &cellDesc);
	void establishNewConnection(CellDescription &cell1, CellDescription &cell2);
	double getDistance(CellDescription &cell1, CellDescription &cell2);

	list<uint64_t> getCellIdsAtPos(IntVector2D const &pos);

	SpaceMetricApi *_metric = nullptr;
	SimulationParameters *_parameters = nullptr;

	map<uint64_t, int> _clusterIndicesByCellIds;
	map<uint64_t, int> _cellIndicesByCellIds;
	map<int, map<int, list<uint64_t>>> _cellMap;
};
