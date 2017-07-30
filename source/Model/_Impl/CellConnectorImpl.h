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
	void updateConnectingCells();
	void reclustering(DataDescription &result);

	void removeConnectionsIfNecessary(CellDescription &cellDesc);
	CellDescription& getCellDescRef(uint64_t cellId);

	SpaceMetricApi *_metric = nullptr;
	SimulationParameters *_parameters = nullptr;

	DataDescription _data;

	map<uint64_t, int> _clusterIndicesByCellIds;
	map<uint64_t, int> _cellIndicesByCellIds;
};
