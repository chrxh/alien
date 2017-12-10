#include <algorithm>

#include "DescriptionHelperImpl.h"

#include "Base/NumberGenerator.h"

#include "Model/Api/SpaceProperties.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Local/Physics.h"


void DescriptionHelperImpl::init(SpaceProperties *metric, SimulationParameters *parameters, NumberGenerator *numberGen)
{
	_metric = metric;
	_parameters = parameters;
	_numberGen = numberGen;
}

void DescriptionHelperImpl::reconnect(DataDescription &data, unordered_set<uint64_t> const &changedCellIds)
{
	if (!data.clusters) {
		return;
	}
	_data = &data;

	updateInternals();
	list<uint64_t> changedAndPresentCellIds = filterPresentCellIds(changedCellIds);
	updateConnectingCells(changedAndPresentCellIds);

	unordered_set<uint64_t> clusterIds;
	for (uint64_t cellId : changedAndPresentCellIds) {
		clusterIds.insert(_navi.clusterIdsByCellIds.at(cellId));
	}
	reclustering(clusterIds);
}

void DescriptionHelperImpl::recluster(DataDescription & data, unordered_set<uint64_t> const & changedClusterIds)
{
	if (!data.clusters) {
		return;
	}
	_data = &data;

	updateInternals();
	reclustering(changedClusterIds);
}

void DescriptionHelperImpl::makeValid(ClusterDescription & cluster)
{
	if (cluster.id == 0) {
		cluster.id = _numberGen->getTag();
	}
	if (cluster.cells) {
		CHECK(cluster.cells->size() == 1);	//only one cell supported

		if (cluster.cells->front().id == 0) {
			cluster.cells->front().id = _numberGen->getTag();
		}
	}
}

void DescriptionHelperImpl::makeValid(ParticleDescription & particle)
{
	if (particle.id == 0) {
		particle.id = _numberGen->getTag();
	}
}

list<uint64_t> DescriptionHelperImpl::filterPresentCellIds(unordered_set<uint64_t> const & cellIds) const
{
	list<uint64_t> result;
	std::copy_if(cellIds.begin(), cellIds.end(), std::back_inserter(result), [&](auto const& cellId) {
		return _navi.cellIds.find(cellId) != _navi.cellIds.end();
	});
	return result;
}

void DescriptionHelperImpl::updateInternals()
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

void DescriptionHelperImpl::updateConnectingCells(list<uint64_t> const &changedCellIds)
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

void DescriptionHelperImpl::reclustering(unordered_set<uint64_t> const& clusterIds)
{
	unordered_set<uint64_t> affectedClusterIndices;
	for (uint64_t clusterId : clusterIds) {
		affectedClusterIndices.insert(_navi.clusterIndicesByClusterIds.at(clusterId));
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

void DescriptionHelperImpl::lookUpCell(uint64_t cellId, ClusterDescription &newCluster, unordered_set<uint64_t> &lookedUpCellIds
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

CellDescription & DescriptionHelperImpl::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
	ClusterDescription &cluster = _data->clusters->at(clusterIndex);
	return cluster.cells->at(cellIndex);
}

void DescriptionHelperImpl::removeConnections(CellDescription &cellDesc)
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

void DescriptionHelperImpl::establishNewConnectionsWithNeighborCells(CellDescription & cellDesc)
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

void DescriptionHelperImpl::establishNewConnection(CellDescription &cell1, CellDescription &cell2) const
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

double DescriptionHelperImpl::getDistance(CellDescription &cell1, CellDescription &cell2) const
{
	auto &pos1 = *cell1.pos;
	auto &pos2 = *cell2.pos;
	auto displacement = pos2 - pos1;
	_metric->correctDisplacement(displacement);
	return displacement.length();
}

list<uint64_t> DescriptionHelperImpl::getCellIdsAtPos(IntVector2D const &pos)
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

namespace
{
	QVector2D calcCenter(vector<CellDescription> const & cells)
	{
		QVector2D result;
		for (auto const& cell : cells) {
			result += *cell.pos;
		}
		result = result / cells.size();
		return result;
	}
}

void DescriptionHelperImpl::setClusterAttributes(ClusterDescription& cluster)
{
	cluster.pos = calcCenter(*cluster.cells);
	cluster.angle = calcAngleBasedOnOldClusters(*cluster.cells);
	auto velocities = calcVelocitiesBasedOnOldClusters(*cluster.cells);
	cluster.vel = velocities.linearVel;
	cluster.angularVel = velocities.angularVel;
	if (auto clusterMetadata = calcMetadataBasedOnOldClusters(*cluster.cells)) {
		cluster.metadata = *clusterMetadata;
	}
}

double DescriptionHelperImpl::calcAngleBasedOnOldClusters(vector<CellDescription> const & cells) const
{
	qreal result = 0.0;
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		result += *_data->clusters->at(clusterIndex).angle;
	}
	result /= cells.size();
	return result;
}

namespace
{
	double calcAngularMass(vector<CellDescription> const & cells)
	{
		QVector2D center = calcCenter(cells);
		double result = 0.0;
		for (auto const& cell : cells) {
			result += (*cell.pos - center).lengthSquared();
		}
		return result;
	}
}

DescriptionHelperImpl::ClusterVelocities DescriptionHelperImpl::calcVelocitiesBasedOnOldClusters(vector<CellDescription> const & cells) const
{
	CHECK(!cells.empty());
	
	ClusterVelocities result;
	if (cells.size() == 1) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cells.front().id);
		int cellIndex = _navi.cellIndicesByCellIds.at(cells.front().id);
		auto const& cluster = _data->clusters->at(clusterIndex);
		result.linearVel = Physics::tangentialVelocity(*cells.front().pos - *cluster.pos, *cluster.vel, *cluster.angularVel);
		return result;
	}

	QVector2D center = calcCenter(cells);
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		int cellIndex = _navi.cellIndicesByCellIds.at(cell.id);
		auto const& cluster = _data->clusters->at(clusterIndex);
		result.linearVel += Physics::tangentialVelocity(*cell.pos - center, *cluster.vel, *cluster.angularVel);
	}
	result.linearVel /= cells.size();

	double angularMomentum = 0.0;
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		int cellIndex = _navi.cellIndicesByCellIds.at(cell.id);
		auto const& cluster = _data->clusters->at(clusterIndex);
		auto cellVel = Physics::tangentialVelocity(*cell.pos - center, *cluster.vel, *cluster.angularVel);
		QVector2D r = *cell.pos - center;
		QVector2D v = cellVel - result.linearVel;
		angularMomentum += Physics::angularMomentum(r, v);
	}
	result.angularVel = Physics::angularVelocity(angularMomentum, calcAngularMass(cells));

	return result;
}

optional<ClusterMetadata> DescriptionHelperImpl::calcMetadataBasedOnOldClusters(vector<CellDescription> const & cells) const
{
	CHECK(!cells.empty());

	map<int, int> clusterCount;
	for (auto const& cell : cells) {
		int clusterId = _navi.clusterIndicesByCellIds.at(cell.id);
		clusterCount[clusterId]++;
	}

	int maxClusterCount = 0;
	int clusterIndexWithMaxCount = 0;
	for (auto const& clusterAndCount : clusterCount) {
		if (clusterAndCount.second > maxClusterCount) {
			clusterIndexWithMaxCount = clusterAndCount.first;
			maxClusterCount = clusterAndCount.second;
		}
	}
	auto clusterWithMaxCount = _data->clusters->at(clusterIndexWithMaxCount);
	return clusterWithMaxCount.metadata;
}
