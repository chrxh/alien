#include "Base/NumberGenerator.h"
#include "ModelBasic/Descriptions.h"

#include "DataConverter.h"

DataConverter::DataConverter(SimulationDataForAccess& cudaData, NumberGenerator* numberGen)
	: _cudaData(cudaData), _numberGen(numberGen)
{}

void DataConverter::add(ClusterDescription const& clusterDesc)
{
	if (!clusterDesc.cells) {
		return;
	}

	ClusterData& cudaCluster = _cudaData.clusters[_cudaData.numClusters++];
	cudaCluster.id = clusterDesc.id == 0 ? _numberGen->getTag() : clusterDesc.id;
	QVector2D clusterPos = clusterDesc.pos ? clusterPos = *clusterDesc.pos : clusterPos = clusterDesc.getClusterPosFromCells();
	cudaCluster.pos = { clusterPos.x(), clusterPos.y() };
	cudaCluster.vel = { clusterDesc.vel->x(), clusterDesc.vel->y() };
	cudaCluster.angle = *clusterDesc.angle;
	cudaCluster.angularVel = *clusterDesc.angularVel;
	unordered_map<uint64_t, CellData*> cellByIds;
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		addCell(cellDesc, clusterDesc, cudaCluster, cellByIds);
	}
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		if (cellDesc.id != 0) {
			resolveConnections(cellDesc, cellByIds, *cellByIds.at(cellDesc.id));
		}
	}

	updateAngularMass(cudaCluster);
}

SimulationDataForAccess DataConverter::getResult() const
{
	return _cudaData;
}

void DataConverter::addCell(CellDescription const& cellDesc, ClusterDescription const& cluster, ClusterData& cudaCluster
	, unordered_map<uint64_t, CellData*>& cellByIds)
{
	CellData& cudaCell = _cudaData.cells[_cudaData.numCells++];
	cudaCell.id = cellDesc.id == 0 ? _numberGen->getTag() : cellDesc.id;
	cudaCell.cluster = &cudaCluster;
	cudaCell.absPos = { cellDesc.pos->x(), cellDesc.pos->y() };
	QVector2D relPos = cellDesc.getPosRelativeTo(cluster);
	cudaCell.relPos = { relPos.x(), relPos.y() };
	cudaCell.energy = *cellDesc.energy;
	if (cellDesc.connectingCells) {
		cudaCell.numConnections = cellDesc.connectingCells->size();
	}
	else {
		cudaCell.numConnections = 0;
	}
	cudaCell.protectionCounter = 0;
	cudaCell.setProtectionCounterForNextTimestep = false;
	cudaCell.alive = true;

	cellByIds.insert_or_assign(cudaCell.id, &cudaCell);
	++_cudaData.numCells;
}

void DataConverter::resolveConnections(CellDescription const & cellToAdd, unordered_map<uint64_t, CellData*> const & cellByIds
	, CellData& cudaCell)
{
	int index = 0;
	if (cellToAdd.connectingCells) {
		for (uint64_t connectingCellId : *cellToAdd.connectingCells) {
			cudaCell.connections[index] = cellByIds.at(connectingCellId);
			++index;
		}
	}
}

void DataConverter::updateAngularMass(ClusterData& cluster)
{
	cluster.angularMass = 0.0;
	for (int i = 0; i < cluster.numCells; ++i) {
		float2 relPos = cluster.cells[i].relPos;
		cluster.angularMass += relPos.x*relPos.x + relPos.y*relPos.y;
	}
}

