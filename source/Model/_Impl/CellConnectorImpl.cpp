#include "CellConnectorImpl.h"

#include "Base/NumberGenerator.h"

#include "Model/Context/SpaceMetricApi.h"
#include "Model/Context/SimulationParameters.h"


void CellConnectorImpl::init(SpaceMetricApi *metric, SimulationParameters *parameters, NumberGenerator *numberGen)
{
	_metric = metric;
	_parameters = parameters;
	_numberGen = numberGen;
}

void CellConnectorImpl::reconnect(DataDescription &data)
{
	updateInternals(data);
	updateConnectingCells(data);
	reclustering(data);

/*
	DataDescription dataNew;
	reclustering(dataNew);
	data = dataNew;
*/
}

void CellConnectorImpl::updateInternals(DataDescription const &data)
{
	_clusterIndicesByCellIds.clear();
	_cellIndicesByCellIds.clear();
	_cellMap.clear();

	int clusterIndex = 0;
	for (auto const &cluster : getUndeletedElements(data.clusters)) {
		int cellIndex = 0;
		for (auto const &cell : getUndeletedElements(cluster.cells)) {
			_clusterIndicesByCellIds[cell.id] = clusterIndex;
			_cellIndicesByCellIds[cell.id] = cellIndex;
			auto const &pos = cell.pos.getValue();
			auto intPos = _metric->correctPositionAndConvertToIntVector(pos);
			_cellMap[intPos.x][intPos.y].push_back(cell.id);
			++cellIndex;
		}
		++clusterIndex;
	}
}


void CellConnectorImpl::updateConnectingCells(DataDescription &data)
{
	for (auto &clusterT : data.clusters) {
		if (clusterT.isDeleted()) { continue; }
		auto &clusterD = clusterT.getValue();
		for (auto &cellT : clusterD.cells) {
			if (cellT.isDeleted()) { continue; }
			auto &cellD = cellT.getValue();
			if (cellD.pos.isModified()) {
				removeConnections(data, cellD);
			}
		}
	}
	for (auto &clusterT : data.clusters) {
		if (clusterT.isDeleted()) { continue; }
		auto &clusterD = clusterT.getValue();
		int cellIndex = 0;
		for (auto &cellT : clusterD.cells) {
			if (cellT.isDeleted()) { continue; }
			auto &cellD = cellT.getValue();
			if (cellD.pos.isModified()) {
				establishNewConnectionsWithNeighborCells(data, cellD);
			}
		}
	}
}

void CellConnectorImpl::reclustering(DataDescription &data)
{
	unordered_set<int> affectedClusterIndices;
	int clusterIndex = 0;
	for (auto &clusterT : data.clusters) {
		if (clusterT.isDeleted()) { continue; }
		auto &clusterD = clusterT.getValue();
		for (auto &cellT : clusterD.cells) {
			if (cellT.isDeleted()) { continue; }
			auto &cellD = cellT.getValue();
			if (cellD.connectingCells.isModified()) {
				affectedClusterIndices.insert(clusterIndex);
			}
		}
		++clusterIndex;
	}
	
	while (!affectedClusterIndices.empty()) {
		int affectedClusterIndex = *affectedClusterIndices.begin();
		unordered_set<int> modifiedClusterIndices = reclusteringSingleClusterAndReturnModifiedClusterIndices(data, affectedClusterIndex);
		for (int modifiedClusterIndex : modifiedClusterIndices) {
			affectedClusterIndices.erase(modifiedClusterIndex);
		}
	}
}

unordered_set<int> CellConnectorImpl::reclusteringSingleClusterAndReturnModifiedClusterIndices(DataDescription &data, int clusterIndex)
{
	auto &clusterT = data.clusters.at(clusterIndex);
	auto &clusterD = clusterT.getValue();

	unordered_set<uint64_t> lookedUpCellIds;
	unordered_set<uint64_t> remainingCellIdsOfCluster;
	for (auto &cellT : clusterD.cells) {
		auto &cellD = cellT.getValue();
		remainingCellIdsOfCluster.insert(cellD.id);
	}

	vector<CellClusterDescription> newClusters;
	bool firstRun = true;
	while (!remainingCellIdsOfCluster.empty()) {
		uint64_t remainingCellIdOfCluster = *remainingCellIdsOfCluster.begin();
		CellClusterDescription newCluster;
		lookUpCell(data, remainingCellIdOfCluster, newCluster, lookedUpCellIds, remainingCellIdsOfCluster);
		if (!newCluster.cells.empty()) {
			if (firstRun) {
				newCluster.id = clusterD.id;
				firstRun = false;
			}
			else {
				newCluster.id = _numberGen->getTag();
			}
			newClusters.push_back(newCluster);
		}
	}

	//TODO: restliche Clusterdaten füllen (pos, ...)
	//ERR: Verbindung lösen und dann neu verbinden klappt noch nicht

	unordered_set<int> discardClusterIndices;
	for (uint64_t lookedUpCellId : lookedUpCellIds) {
		discardClusterIndices.insert(_clusterIndicesByCellIds.at(lookedUpCellId));
	}
	auto result = discardClusterIndices;

	for (auto &newCluster : newClusters) {
		if (!discardClusterIndices.empty()) {
			int discardClusterIndex = *discardClusterIndices.begin();
			//TODO: Tracker der Cells in newCluster setzen... Methode in CellClusterDescription?
			data.clusters[discardClusterIndex].setValue(newCluster);
			discardClusterIndices.erase(discardClusterIndex);
		}
		else {
			data.addCellCluster(newCluster);
		}
	}
	for (auto &discardClusterIndex : discardClusterIndices) {
		data.clusters[discardClusterIndex].setAsDeleted();
		for (auto &cellT : data.clusters[discardClusterIndex]->cells) {
			cellT.setAsDeleted();
		}
	}

	return result;
}

void CellConnectorImpl::lookUpCell(DataDescription &data, uint64_t cellId, CellClusterDescription &newCluster
	, unordered_set<uint64_t> &lookedUpCellIds, unordered_set<uint64_t> &remainingCellIds)
{
	if (lookedUpCellIds.find(cellId) != lookedUpCellIds.end()) {
		return;
	}
	
	lookedUpCellIds.insert(cellId);
	remainingCellIds.erase(cellId);

	auto &cell = getCellDescRef(data, cellId);
	newCluster.addCell(cell);

	if (cell.connectingCells.isInitialized()) {
		for (uint64_t connectingCellId : cell.connectingCells.getValue()) {
			lookUpCell(data, connectingCellId, newCluster, lookedUpCellIds, remainingCellIds);
		}
	}
}

CellDescription & CellConnectorImpl::getCellDescRef(DataDescription &data, uint64_t cellId)
{
	int clusterIndex = _clusterIndicesByCellIds.at(cellId);
	int cellIndex = _cellIndicesByCellIds.at(cellId);
	CellClusterDescription &clusterDesc = data.clusters.at(clusterIndex).getValue();
	return clusterDesc.cells[cellIndex].getValue();
}

void CellConnectorImpl::removeConnections(DataDescription &data, CellDescription &cellDesc)
{
	if (cellDesc.connectingCells.isInitialized()) {
		auto &connectingCellIds = cellDesc.connectingCells.getValue();
		for (uint64_t connectingCellId : connectingCellIds) {
			auto &connectingCell = getCellDescRef(data, connectingCellId);
			auto &connectingCellConnections = connectingCell.connectingCells.getValue();
			connectingCellConnections.remove(cellDesc.id);
		}
		cellDesc.connectingCells.reset();
	}
}

void CellConnectorImpl::establishNewConnectionsWithNeighborCells(DataDescription & data, CellDescription & cellDesc)
{
	int r = static_cast<int>(std::ceil(_parameters->cellMaxDistance));
	IntVector2D pos = cellDesc.pos.getValue();
	for(int dx = -r; dx <= r; ++dx) {
		for (int dy = -r; dy <= r; ++dy) {
			IntVector2D scanPos = { pos.x + dx, pos.y + dy };
			_metric->correctPosition(scanPos);
			auto cellIds = getCellIdsAtPos(scanPos);
			for (uint64_t cellId : cellIds) {
				establishNewConnection(cellDesc, getCellDescRef(data, cellId));
			}
		}
	}
}

void CellConnectorImpl::establishNewConnection(CellDescription &cell1, CellDescription &cell2)
{
	if (cell1.id == cell2.id) {
		return;
	}
	if (getDistance(cell1, cell2) > _parameters->cellMaxDistance) {
		return;
	}
	if (cell1.connectingCells.getValueOrDefault().size() >= cell1.maxConnections.getValueOrDefault()
		|| cell2.connectingCells.getValueOrDefault().size() >= cell2.maxConnections.getValueOrDefault()) {
		return;
	}
	if (!cell1.connectingCells.isInitialized()) {
		cell1.connectingCells.setValue(list<uint64_t>());
	}
	if (!cell2.connectingCells.isInitialized()) {
		cell2.connectingCells.setValue(list<uint64_t>());
	}
	auto &connections1 = cell1.connectingCells.getValue();
	auto &connections2 = cell2.connectingCells.getValue();
	connections1.push_back(cell2.id);
	connections2.push_back(cell1.id);
}

double CellConnectorImpl::getDistance(CellDescription &cell1, CellDescription &cell2)
{
	auto &pos1 = cell1.pos.getValue();
	auto &pos2 = cell2.pos.getValue();
	auto displacement = pos2 - pos1;
	_metric->correctDisplacement(displacement);
	return displacement.length();
}

list<uint64_t> CellConnectorImpl::getCellIdsAtPos(IntVector2D const &pos)
{
	auto xIter = _cellMap.find(pos.x);
	if (xIter != _cellMap.end()) {
		unordered_map<int, list<uint64_t>> &mapRemainder = xIter->second;
		auto yIter = mapRemainder.find(pos.y);
		if (yIter != mapRemainder.end()) {
			return yIter->second;
		}
	}
	return list<uint64_t>();
}
