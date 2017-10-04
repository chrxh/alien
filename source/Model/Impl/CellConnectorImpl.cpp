#include <algorithm>

#include "CellConnectorImpl.h"

#include "Base/NumberGenerator.h"

#include "Model/Api/SpaceMetric.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Local/Physics.h"


void CellConnectorImpl::init(SpaceMetric *metric, SimulationParameters *parameters, NumberGenerator *numberGen)
{
	_metric = metric;
	_parameters = parameters;
	_numberGen = numberGen;
}

void CellConnectorImpl::reconnect(DataDescription &data, list<uint64_t> const &changedCellIds)
{
	if (!data.clusters) {
		return;
	}
	_data = &data;
	updateInternals();
	list<uint64_t> changedAndPresentCellIds = filterPresentCellIds(changedCellIds);
	updateConnectingCells(changedAndPresentCellIds);
	reclustering(changedAndPresentCellIds);
}

list<uint64_t> CellConnectorImpl::filterPresentCellIds(list<uint64_t> const & cellIds) const
{
	list<uint64_t> result;
	std::copy_if(cellIds.begin(), cellIds.end(), std::back_inserter(result), [&](auto const& cellId) {
		return _navi.cellIds.find(cellId) != _navi.cellIds.end();
	});
	return result;
}

void CellConnectorImpl::updateInternals()
{
	_navi.update(*_data);
	_cellMap.clear();

	for (auto const &cluster : *_data->clusters) {
		for (auto const &cell : *cluster.cells) {
			auto const &pos = *cell.pos;
			auto intPos = _metric->correctPositionAndConvertToIntVector(pos);
			_cellMap[intPos.x][intPos.y].push_back(cell.id);
		}
	}
}

void CellConnectorImpl::updateConnectingCells(list<uint64_t> const &changedCellIds)
{
	for (uint64_t changedCellId : changedCellIds) {
		auto &cell = getCellDescRef(changedCellId);
		removeConnections(cell);
	}

	for (uint64_t changedCellId : changedCellIds) {
		auto &cell = getCellDescRef(changedCellId);
		establishNewConnectionsWithNeighborCells(cell);
	}
}

void CellConnectorImpl::reclustering(list<uint64_t> const &changedCellIds)
{
	unordered_set<int> affectedClusterIndices;
	for (uint64_t lookedUpCellId : changedCellIds) {
		affectedClusterIndices.insert(_navi.clusterIndicesByCellIds.at(lookedUpCellId));
	}

	vector<ClusterDescription> newClusters;
	unordered_set<uint64_t> remainingCellIds;
	for (int affectedClusterIndex : affectedClusterIndices) {
		for (auto &cell : *_data->clusters->at(affectedClusterIndex).cells) {
			remainingCellIds.insert(cell.id);
		}
	}

	unordered_set<uint64_t> lookedUpCellIds;

	while (!remainingCellIds.empty()) {
		ClusterDescription newCluster;
		lookUpCell(*remainingCellIds.begin(), newCluster, lookedUpCellIds, remainingCellIds);
		if (newCluster.cells && !newCluster.cells->empty()) {
			newCluster.id = _numberGen->getTag();
			setClusterAttributes(newCluster);
			newClusters.push_back(newCluster);
		}
	}

	unordered_set<int> discardedClusterIndices;
	for (uint64_t lookedUpCellId : lookedUpCellIds) {
		discardedClusterIndices.insert(_navi.clusterIndicesByCellIds.at(lookedUpCellId));
	}

	for (int clusterIndex = 0; clusterIndex < _data->clusters->size(); ++clusterIndex) {
		if (discardedClusterIndices.find(clusterIndex) == discardedClusterIndices.end()) {
			newClusters.emplace_back(_data->clusters->at(clusterIndex));
		}
	}

	_data->clusters = newClusters;
}

void CellConnectorImpl::lookUpCell(uint64_t cellId, ClusterDescription &newCluster, unordered_set<uint64_t> &lookedUpCellIds
	, unordered_set<uint64_t> &remainingCellIds)
{
	if (lookedUpCellIds.find(cellId) != lookedUpCellIds.end()) {
		return;
	}
	
	lookedUpCellIds.insert(cellId);
	remainingCellIds.erase(cellId);

	auto &cell = getCellDescRef(cellId);
	newCluster.addCell(cell);

	if (cell.connectingCells) {
		for (uint64_t connectingCellId : *cell.connectingCells) {
			lookUpCell(connectingCellId, newCluster, lookedUpCellIds, remainingCellIds);
		}
	}
}

CellDescription & CellConnectorImpl::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
	ClusterDescription &cluster = _data->clusters->at(clusterIndex);
	return cluster.cells->at(cellIndex);
}

void CellConnectorImpl::removeConnections(CellDescription &cellDesc)
{
	if (cellDesc.connectingCells) {
		auto &connectingCellIds = *cellDesc.connectingCells;
		for (uint64_t connectingCellId : connectingCellIds) {
			auto &connectingCell = getCellDescRef(connectingCellId);
			auto &connectingCellConnections = *connectingCell.connectingCells;
			connectingCellConnections.remove(cellDesc.id);
		}
		cellDesc.connectingCells = list<uint64_t>();
	}
}

void CellConnectorImpl::establishNewConnectionsWithNeighborCells(CellDescription & cellDesc)
{
	int r = static_cast<int>(std::ceil(_parameters->cellMaxDistance));
	IntVector2D pos = *cellDesc.pos;
	for(int dx = -r; dx <= r; ++dx) {
		for (int dy = -r; dy <= r; ++dy) {
			IntVector2D scanPos = { pos.x + dx, pos.y + dy };
			_metric->correctPosition(scanPos);
			auto cellIds = getCellIdsAtPos(scanPos);
			for (uint64_t cellId : cellIds) {
				establishNewConnection(cellDesc, getCellDescRef(cellId));
			}
		}
	}
}

void CellConnectorImpl::establishNewConnection(CellDescription &cell1, CellDescription &cell2) const
{
	if (cell1.id == cell2.id) {
		return;
	}
	if (getDistance(cell1, cell2) > _parameters->cellMaxDistance) {
		return;
	}
	if (cell1.connectingCells.get_value_or({}).size() >= cell1.maxConnections.get_value_or(0)
		|| cell2.connectingCells.get_value_or({}).size() >= cell2.maxConnections.get_value_or(0)) {
		return;
	}
	if (!cell1.connectingCells) {
		cell1.connectingCells = list<uint64_t>();
	}
	if (!cell2.connectingCells) {
		cell2.connectingCells = list<uint64_t>();
	}
	auto &connections1 = *cell1.connectingCells;
	auto &connections2 = *cell2.connectingCells;
	if (std::find(connections1.begin(), connections1.end(), cell2.id) == connections1.end()) {
		connections1.push_back(cell2.id);
		connections2.push_back(cell1.id);
	}
}

double CellConnectorImpl::getDistance(CellDescription &cell1, CellDescription &cell2) const
{
	auto &pos1 = *cell1.pos;
	auto &pos2 = *cell2.pos;
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

void CellConnectorImpl::setClusterAttributes(ClusterDescription& cluster)
{
	QVector2D center;
	for (auto const& cell : *cluster.cells) {
		center += *cell.pos;
	}
	center = center / cluster.cells->size();
	cluster.pos = center;
	cluster.angle = calcAngleBasedOnOldClusters(*cluster.cells);
	auto velocities = calcVelocitiesBasedOnOldClusters(*cluster.cells);
	cluster.vel = velocities.linearVel;
	cluster.angularVel = velocities.angularVel;
}

double CellConnectorImpl::calcAngleBasedOnOldClusters(vector<CellDescription> const & cells) const
{
	qreal result = 0.0;
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		result += *_data->clusters->at(clusterIndex).angle;
	}
	result /= cells.size();
	return result;
}

CellConnectorImpl::ClusterVelocities CellConnectorImpl::calcVelocitiesBasedOnOldClusters(vector<CellDescription> const & cells) const
{
	ClusterVelocities result;
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		int cellIndex = _navi.cellIndicesByCellIds.at(cell.id);
		auto const& cluster = _data->clusters->at(clusterIndex);
		auto const& cell = cluster.cells->at(cellIndex);
		result.linearVel += Physics::tangentialVelocity(*cell.pos - *cluster.pos, *cluster.vel, *cluster.angularVel);
	}
	result.linearVel /= cells.size();
	return result;
}
