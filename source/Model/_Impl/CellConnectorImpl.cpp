#include "CellConnectorImpl.h"

void CellConnectorImpl::reconnect(DataDescription &data)
{
	updateInternals(data);
	updateConnectingCells(data);

	DataDescription dataNew;
	reclustering(data, dataNew);
	data = dataNew;
}

void CellConnectorImpl::updateInternals(DataDescription const &data)
{
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

void CellConnectorImpl::updateConnectingCells(DataDescription &data)
{
	for (auto &clusterT : data.clusters) {
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

void CellConnectorImpl::reclustering(DataDescription const & dataInput, DataDescription & dataOutput)
{
}

void CellConnectorImpl::removeConnectionsIfNecessary(CellDescription & cellDesc) const
{
	vector<uint64_t> connectingCells = cellDesc.connectingCells.getValueOrDefault();
	for (uint64_t connectingCell : connectingCells) {

	}
}

