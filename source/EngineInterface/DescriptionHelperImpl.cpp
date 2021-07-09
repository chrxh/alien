#include <algorithm>

#include <Base/DebugMacros.h>
#include "Base/NumberGenerator.h"

#include "DescriptionHelperImpl.h"

#include "SpaceProperties.h"
#include "SimulationParameters.h"
#include "SimulationContext.h"
#include "Physics.h"


void DescriptionHelperImpl::init(SimulationContext* context)
{
	_metric = context->getSpaceProperties();
	_parameters = context->getSimulationParameters();
	_numberGen = context->getNumberGenerator();
}

void DescriptionHelperImpl::reconnect(DataDescription &data, DataDescription& orgData, unordered_set<uint64_t> const& idsOfChangedCells)
{
    TRY;
	if (!data.clusters) {
		return;
	}
	_data = &data;
	_origData = &orgData;

	updateInternals();
	list<uint64_t> changedAndPresentCellIds = filterPresentCellIds(idsOfChangedCells);
	updateConnectingCells(changedAndPresentCellIds);

	unordered_set<uint64_t> clusterIds;
	for (uint64_t cellId : changedAndPresentCellIds) {
		clusterIds.insert(_navi.clusterIdsByCellIds.at(cellId));
	}
	reclustering(clusterIds);
    CATCH;
}

void DescriptionHelperImpl::recluster(DataDescription & data, unordered_set<uint64_t> const & idsOfChangedClusters)
{
    TRY;
    if (!data.clusters) {
		return;
	}
	_data = &data;
	_origData = &data;

	updateInternals();
	reclustering(idsOfChangedClusters);
    CATCH;
}

void DescriptionHelperImpl::makeValid(DataDescription & data)
{
    TRY;
    if (data.clusters) {
        for (auto& cluster : *data.clusters) {
            makeValid(cluster);
        }
    }
    if (data.particles) {
        for (auto& particle : *data.particles) {
            makeValid(particle);
        }
    }
    CATCH;
}

void DescriptionHelperImpl::makeValid(ClusterDescription & cluster)
{
    TRY;
    cluster.id = _numberGen->getId();
	if (cluster.cells) {
		unordered_map<uint64_t, uint64_t> newByOldIds;
		for (auto & cell : *cluster.cells) {
			uint64_t newId = _numberGen->getId();
			newByOldIds.insert_or_assign(cell.id, newId);
			cell.id = newId;
		}

		for (auto & cell : *cluster.cells) {
			if (cell.connections) {
                for (auto& connection : *cell.connections) {
                    connection.cellId = newByOldIds.at(connection.cellId);
				}
			}
		}
	}
    CATCH;
}

void DescriptionHelperImpl::makeValid(ParticleDescription & particle)
{
    TRY;
	particle.id = _numberGen->getId();
    CATCH;
}

void DescriptionHelperImpl::duplicate(DataDescription& data, IntVector2D const& origSize, IntVector2D const& size)
{
    TRY;
    DataDescription result;

    for (int incX = 0; incX < size.x; incX += origSize.x) {
        for (int incY = 0; incY < size.y; incY += origSize.y) {
            if (data.clusters) {
                for (auto cluster : *data.clusters) {
                    auto origPos = *cluster.pos;
                    cluster.pos = QVector2D{ origPos.x() + incX, origPos.y() + incY };
                    if (cluster.pos->x() < size.x && cluster.pos->y() < size.y) {
                        if (cluster.cells) {
                            for (auto& cell : *cluster.cells) {
                                auto origPos = *cell.pos;
                                cell.pos = QVector2D{ origPos.x() + incX, origPos.y() + incY };
                            }
                        }
                        result.addCluster(cluster);
                    }
                }
            }
            if (data.particles) {
                for (auto particle : *data.particles) {
                    auto origPos = *particle.pos;
                    particle.pos = QVector2D{ origPos.x() + incX, origPos.y() + incY };
                    if (particle.pos->x() < size.x && particle.pos->y() < size.y) {
                        result.addParticle(particle);
                    }
                }
            }
        }
    }
    data = result;
    CATCH;
}

list<uint64_t> DescriptionHelperImpl::filterPresentCellIds(unordered_set<uint64_t> const & cellIds) const
{
    TRY;
	list<uint64_t> result;
	std::copy_if(cellIds.begin(), cellIds.end(), std::back_inserter(result), [&](auto const& cellId) {
		return _navi.cellIds.find(cellId) != _navi.cellIds.end();
	});
	return result;
    CATCH;
}

void DescriptionHelperImpl::updateInternals()
{
    TRY;
    _navi.update(*_data);
	_origNavi.update(*_origData);

	_cellMap.clear();
	for (auto const &cluster : *_data->clusters) {
		for (auto const &cell : *cluster.cells) {
			auto const &pos = *cell.pos;
			auto intPos = _metric->convertToIntVector(pos);
			_cellMap[intPos.x][intPos.y].push_back(cell.id);
		}
	}
    CATCH;
}

void DescriptionHelperImpl::updateConnectingCells(list<uint64_t> const &changedCellIds)
{
    TRY;
    for (uint64_t changedCellId : changedCellIds) {
		auto &cell = getCellDescRef(changedCellId);
		removeConnections(cell);
	}

	for (uint64_t changedCellId : changedCellIds) {
		auto &cell = getCellDescRef(changedCellId);
		establishNewConnectionsWithNeighborCells(cell);
	}
    CATCH;
}

void DescriptionHelperImpl::reclustering(unordered_set<uint64_t> const& clusterIds)
{
    TRY;
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
			newCluster.id = _numberGen->getId();
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
    CATCH;
}

void DescriptionHelperImpl::lookUpCell(uint64_t cellId, ClusterDescription &newCluster, unordered_set<uint64_t> &lookedUpCellIds
	, unordered_set<uint64_t> &remainingCellIds)
{
    TRY;
    if (lookedUpCellIds.find(cellId) != lookedUpCellIds.end()) {
		return;
	}
	
	lookedUpCellIds.insert(cellId);
	remainingCellIds.erase(cellId);

	auto &cell = getCellDescRef(cellId);
	newCluster.addCell(cell);

	if (cell.connections) {
        for (auto const& connection : *cell.connections) {
            lookUpCell(connection.cellId, newCluster, lookedUpCellIds, remainingCellIds);
		}
	}
    CATCH;
}

CellDescription & DescriptionHelperImpl::getCellDescRef(uint64_t cellId)
{
    TRY;
    int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
	ClusterDescription &cluster = _data->clusters->at(clusterIndex);
	return cluster.cells->at(cellIndex);
    CATCH;
}

void DescriptionHelperImpl::removeConnections(CellDescription &cellDesc)
{
    TRY;
    if (cellDesc.connections) {
        auto& connections = *cellDesc.connections;
        for (auto const& connection : connections) {
            auto& connectingCell = getCellDescRef(connection.cellId);
			auto& connectingCellConnections = *connectingCell.connections;
            auto it = std::find_if(
                connectingCellConnections.begin(),
                connectingCellConnections.end(),
                [&cellDesc](auto const& connection) {
                    return connection.cellId == cellDesc.id;
				});
            if (it != connectingCellConnections.end()) {
                connectingCellConnections.erase(it);
            }
            //			connectingCellConnections.remove(cellDesc.id);
		}
		cellDesc.connections = list<ConnectionDescription>();
	}
    CATCH;
}

void DescriptionHelperImpl::establishNewConnectionsWithNeighborCells(CellDescription & cellDesc)
{
    TRY;
    int r = static_cast<int>(std::ceil(_parameters.cellMaxDistance));
	IntVector2D pos = *cellDesc.pos;
	for(int dx = -r; dx <= r; ++dx) {
		for (int dy = -r; dy <= r; ++dy) {
			IntVector2D scanPos = { pos.x + dx, pos.y + dy };
			auto cellIds = getCellIdsAtPos(scanPos);
			for (uint64_t cellId : cellIds) {
				establishNewConnection(cellDesc, getCellDescRef(cellId));
			}
		}
	}
    CATCH;
}

void DescriptionHelperImpl::establishNewConnection(CellDescription &cell1, CellDescription &cell2) const
{
    TRY;
    if (cell1.id == cell2.id) {
		return;
	}
	if (getDistance(cell1, cell2) > _parameters.cellMaxDistance) {
		return;
	}
	if (cell1.connections.get_value_or({}).size() >= cell1.maxConnections.get_value_or(0)
        || cell2.connections.get_value_or({}).size() >= cell2.maxConnections.get_value_or(0)) {
		return;
	}
    if (!cell1.connections) {
        cell1.connections = list<ConnectionDescription>();
	}
    if (!cell2.connections) {
        cell2.connections = list<ConnectionDescription>();
	}
    auto& connections1 = *cell1.connections;
    auto& connections2 = *cell2.connections;
    if (std::find_if(
            connections1.begin(),
            connections1.end(),
            [&cell2](auto const& connection) { return connection.cellId == cell2.id; })
        == connections1.end()) {

		ConnectionDescription connection1;
        connection1.cellId = cell1.id;
        ConnectionDescription connection2;
        connection2.cellId = cell2.id;
        connections1.emplace_back(connection2);
        connections2.emplace_back(connection1);
	}
    CATCH;
}

double DescriptionHelperImpl::getDistance(CellDescription &cell1, CellDescription &cell2) const
{
    TRY;
    auto& pos1 = *cell1.pos;
	auto &pos2 = *cell2.pos;
	auto displacement = pos2 - pos1;
	return displacement.length();
    CATCH;
}

list<uint64_t> DescriptionHelperImpl::getCellIdsAtPos(IntVector2D const &pos)
{
    TRY;
    auto xIter = _cellMap.find(pos.x);
	if (xIter != _cellMap.end()) {
		unordered_map<int, list<uint64_t>> &mapRemainder = xIter->second;
		auto yIter = mapRemainder.find(pos.y);
		if (yIter != mapRemainder.end()) {
			return yIter->second;
		}
	}
	return list<uint64_t>();
    CATCH;
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
    TRY;
    cluster.pos = calcCenter(*cluster.cells);
	cluster.angle = calcAngleBasedOnOrigClusters(*cluster.cells);
	auto velocities = calcVelocitiesBasedOnOrigClusters(*cluster.cells);
	double v = velocities.linear.length();
	cluster.vel = velocities.linear;
	cluster.angularVel = velocities.angular;
	if (auto clusterMetadata = calcMetadataBasedOnOrigClusters(*cluster.cells)) {
		cluster.metadata = *clusterMetadata;
	}
    CATCH;
}

double DescriptionHelperImpl::calcAngleBasedOnOrigClusters(vector<CellDescription> const & cells) const
{
    TRY;
    qreal result = 0.0;
	for (auto const& cell : cells) {
		int clusterIndex = _navi.clusterIndicesByCellIds.at(cell.id);
		result += *_data->clusters->at(clusterIndex).angle;
	}
	result /= cells.size();
	return result;
    CATCH;
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

Physics::Velocities DescriptionHelperImpl::calcVelocitiesBasedOnOrigClusters(vector<CellDescription> const & cells) const
{
    TRY;
    CHECK(!cells.empty());
	
	Physics::Velocities result{ QVector2D(), 0.0 };
	if (cells.size() == 1) {
		auto cell = cells.front();
		if (_origNavi.clusterIndicesByCellIds.find(cell.id) == _origNavi.clusterIndicesByCellIds.end()
			|| _origNavi.cellIndicesByCellIds.find(cell.id) == _origNavi.cellIndicesByCellIds.end()) {
			return result;
		}
		int clusterIndex = _origNavi.clusterIndicesByCellIds.at(cell.id);
		int cellIndex = _origNavi.cellIndicesByCellIds.at(cell.id);
		auto const& origCluster = _origData->clusters->at(clusterIndex);
		auto const& origCell = origCluster.cells->at(cellIndex);
		result.linear= Physics::tangentialVelocity(*origCell.pos - *origCluster.pos, { *origCluster.vel, *origCluster.angularVel });
		return result;
	}

	unordered_map<uint64_t, QVector2D> cellVel;
	for (auto const& cell : cells) {
		if (_origNavi.clusterIndicesByCellIds.find(cell.id) == _origNavi.clusterIndicesByCellIds.end()
			|| _origNavi.cellIndicesByCellIds.find(cell.id) == _origNavi.cellIndicesByCellIds.end()) {
			return result;
		}
		int clusterIndex = _origNavi.clusterIndicesByCellIds.at(cell.id);
		int cellIndex = _origNavi.cellIndicesByCellIds.at(cell.id);
		auto const& origCluster = _origData->clusters->at(clusterIndex);
		auto const& origCell = origCluster.cells->at(cellIndex);
		cellVel.insert_or_assign(cell.id, Physics::tangentialVelocity(*origCell.pos - *origCluster.pos, { *origCluster.vel, *origCluster.angularVel }));
		result.linear += cellVel.at(cell.id);
	}
	result.linear /= cells.size();

	QVector2D center = calcCenter(cells);
	double angularMomentum = 0.0;
	for (auto const& cell : cells) {
		QVector2D r = *cell.pos - center;
		QVector2D v = cellVel.at(cell.id) - result.linear;
		angularMomentum += Physics::angularMomentum(r, v);
	}
	result.angular = Physics::angularVelocity(angularMomentum, calcAngularMass(cells));

	return result;
    CATCH;
}

boost::optional<ClusterMetadata> DescriptionHelperImpl::calcMetadataBasedOnOrigClusters(vector<CellDescription> const & cells) const
{
    TRY;
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
    CATCH;
}
