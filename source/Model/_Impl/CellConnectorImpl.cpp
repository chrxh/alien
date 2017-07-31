#include "CellConnectorImpl.h"

#include "Model/Context/SpaceMetricApi.h"
#include "Model/Context/SimulationParameters.h"

void CellConnectorImpl::init(SpaceMetricApi * metric, SimulationParameters * parameters)
{
	_metric = metric;
	_parameters = parameters;
}

void CellConnectorImpl::reconnect(DataDescription &data)
{
	updateInternals(data);
	updateConnectingCells();

	data = _data;
/*
	DataDescription dataNew;
	reclustering(dataNew);
	data = dataNew;
*/
}

void CellConnectorImpl::updateInternals(DataDescription const &data)
{
	_data = data;
	_clusterIndicesByCellIds.clear();
	_cellIndicesByCellIds.clear();

	int clusterIndex = 0;
	for (auto const &clusterT : data.clusters) {
		auto const &clusterD = clusterT.getValue();
		int cellIndex = 0;
		for (auto const &cellT : clusterD.cells) {
			auto const &cellD = cellT.getValue();
			_clusterIndicesByCellIds[cellD.id] = clusterIndex;
			_cellIndicesByCellIds[cellD.id] = cellIndex;
			auto const &pos = cellD.pos.getValue();
			auto intPos = _metric->correctPositionAndConvertToIntVector(pos);
			_cellMap[intPos.x][intPos.x].push_back(cellD.id);
			++cellIndex;
		}
		++clusterIndex;
	}
}


void CellConnectorImpl::updateConnectingCells()
{
	for (auto &clusterT : _data.clusters) {
		auto &clusterD = clusterT.getValue();
		int cellIndex = 0;
		for (auto &cellT : clusterD.cells) {
			auto &cellD = cellT.getValue();
			if (cellD.pos.isModified()) {
				removeConnections(cellD);
//				lookingForNewConnections(cellD)
			}
		}
	}
}

void CellConnectorImpl::reclustering(DataDescription &result)
{
}

CellDescription & CellConnectorImpl::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _clusterIndicesByCellIds.at(cellId);
	int cellIndex = _cellIndicesByCellIds.at(cellId);
	CellClusterDescription &clusterDesc = _data.clusters[clusterIndex].getValue();
	return clusterDesc.cells[cellIndex].getValue();
}

void CellConnectorImpl::removeConnections(CellDescription &cellDesc)
{
	if (cellDesc.connectingCells.isInitialized()) {
		auto &connectingCellIds = cellDesc.connectingCells.getValue();
		for (uint64_t connectingCellId : connectingCellIds) {
			auto &connectingCell = getCellDescRef(connectingCellId);
			auto &connectingCellConnections = connectingCell.connectingCells.getValue();
			connectingCellConnections.remove(cellDesc.id);
		}
		cellDesc.connectingCells.reset();
	}
}
