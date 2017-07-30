#pragma once

#include "Model/CellConnector.h"

class CellConnectorImpl
	: public CellConnector
{
	Q_OBJECT
public:
	CellConnectorImpl(QObject *parent = nullptr) : CellConnector(parent) { }
	virtual ~CellConnectorImpl() = default;

	virtual void reconnect(DataDescription &data) override;

private:
	void updateInternals(DataDescription const &data);
	void updateConnectingCells(DataDescription &data);
	void reclustering(DataDescription const &dataInput, DataDescription &dataOutput);

	void removeConnectionsIfNecessary(CellDescription &cellDesc) const;

	map<uint64_t, int> _clusterIndicesByCellIds;
	map<uint64_t, int> _cellIndicesByCellIds;
};
