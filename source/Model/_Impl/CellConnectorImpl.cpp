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

	DataDescription dataNew;
	reclustering(dataNew);
	data = dataNew;
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
				removeConnectionsIfNecessary(cellD);
			}
		}
	}
}

void CellConnectorImpl::reclustering(DataDescription &result)
{
}

void CellConnectorImpl::removeConnectionsIfNecessary(CellDescription &cellDesc)
{
	auto cellPos = cellDesc.pos.getValue();
	auto const& connectingCellIds = cellDesc.connectingCells.getValueOrDefault();
	if (connectingCellIds.empty()) {
		return;
	}
	vector<uint64_t> connectingCellIdsNew;
	for (uint64_t connectingCellId : connectingCellIds) {
		auto const &connectingCell = getCellDescRef(connectingCellId);
		auto connectingCellPos = connectingCell.pos.getValue();
		auto displacement = connectingCellPos - cellPos;
		_metric->correctDisplacement(displacement);
		if (displacement.length() <= _parameters->cellMaxDistance) {
			connectingCellIdsNew.push_back(connectingCellId);
		}
	}
	cellDesc.connectingCells.setValue(connectingCellIdsNew);
}

CellDescription & CellConnectorImpl::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _clusterIndicesByCellIds.at(cellId);
	int cellIndex = _cellIndicesByCellIds.at(cellId);
	CellClusterDescription &clusterDesc = _data.clusters[clusterIndex].getValue();
	return clusterDesc.cells[cellIndex].getValue();
}

