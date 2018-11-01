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
	cudaCluster.numCells = clusterDesc.cells ? clusterDesc.cells->size() : 0;
	unordered_map<uint64_t, CellData*> cellByIds;
	bool firstIndex = true;
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		addCell(cellDesc, clusterDesc, cudaCluster, cellByIds);
		if (firstIndex) {
			cudaCluster.cells = cellByIds.begin()->second;
			firstIndex = false;
		}
	}
	for (CellDescription const& cellDesc : *clusterDesc.cells) {
		if (cellDesc.id != 0) {
			resolveConnections(cellDesc, cellByIds, *cellByIds.at(cellDesc.id));
		}
	}

	updateAngularMass(cudaCluster);
}

void DataConverter::add(ParticleDescription const & particleDesc)
{
	ParticleData& cudaParticle = _cudaData.particles[_cudaData.numParticles++];
	cudaParticle.id = cudaParticle.id == 0 ? _numberGen->getTag() : particleDesc.id;
	cudaParticle.pos = { particleDesc.pos->x(), particleDesc.pos->y() };
	cudaParticle.vel = { particleDesc.vel->x(), particleDesc.vel->y() };
	cudaParticle.energy = *particleDesc.energy;
}

void DataConverter::del(uint64_t clusterId)
{
	_clusterIdsToDelete.insert(clusterId);
}

void DataConverter::finalize()
{
	if (_clusterIdsToDelete.empty()) {
		return;
	}

	//delete clusters
	std::unordered_set<uint64_t> cellIdsToDelete;
	std::unordered_map<ClusterData*, ClusterData*> newByOldClusterData;
	int clusterIndexCopyOffset = 0;
	for (int clusterIndex = 0; clusterIndex < _cudaData.numClusters; ++clusterIndex) {
		ClusterData& cluster = _cudaData.clusters[clusterIndex];
		uint64_t clusterId = cluster.id;
		if (_clusterIdsToDelete.find(clusterId) != _clusterIdsToDelete.end()) {
			++clusterIndexCopyOffset;
			for (int cellIndex = 0; cellIndex < cluster.numCells; ++cellIndex) {
				cellIdsToDelete.insert(cluster.cells[cellIndex].id);
			}
		}
		else if (clusterIndexCopyOffset > 0) {
			newByOldClusterData.insert_or_assign(&cluster, &_cudaData.clusters[clusterIndex - clusterIndexCopyOffset]);
			_cudaData.clusters[clusterIndex - clusterIndexCopyOffset] = cluster;
		}
	}
	_cudaData.numClusters -= clusterIndexCopyOffset;

	//delete cells
	int cellIndexCopyOffset = 0;
	std::unordered_map<CellData*, CellData*> newByOldCellData;
	for (int cellIndex = 0; cellIndex < _cudaData.numCells; ++cellIndex) {
		CellData& cell = _cudaData.cells[cellIndex];
		uint64_t cellId = cell.id;
		if (cellIdsToDelete.find(cellId) != cellIdsToDelete.end()) {
			++cellIndexCopyOffset;
		}
		else if (cellIndexCopyOffset > 0) {
			newByOldCellData.insert_or_assign(&cell, &_cudaData.cells[cellIndex - cellIndexCopyOffset]);
			_cudaData.cells[cellIndex - cellIndexCopyOffset] = cell;
		}
	}
	_cudaData.numCells -= cellIndexCopyOffset;

	//adjust cell and cluster pointers
	for (int clusterIndex = 0; clusterIndex < _cudaData.numClusters; ++clusterIndex) {
		ClusterData& cluster = _cudaData.clusters[clusterIndex];
		auto it = newByOldCellData.find(cluster.cells);
		if (it != newByOldCellData.end()) {
			cluster.cells = it->second;
		}
	}
	for (int cellIndex = 0; cellIndex < _cudaData.numCells; ++cellIndex) {
		CellData& cell = _cudaData.cells[cellIndex];
		auto it = newByOldClusterData.find(cell.cluster);
		if (it != newByOldClusterData.end()) {
			cell.cluster = it->second;
		}
		for (int connectionIndex = 0; connectionIndex < cell.numConnections; ++connectionIndex) {
			auto it = newByOldCellData.find(cell.connections[connectionIndex]);
			if (it != newByOldCellData.end()) {
				cell.connections[connectionIndex] = it->second;
			}
		}
	}
}

SimulationDataForAccess DataConverter::getGpuData() const
{
	return _cudaData;
}

DataDescription DataConverter::getDataDescription(IntRect const& requiredRect) const
{
	DataDescription result;
	list<uint64_t> connectingCellIds;
	for (int i = 0; i < _cudaData.numClusters; ++i) {
		ClusterData const& cluster = _cudaData.clusters[i];
		if (requiredRect.isContained({ int(cluster.pos.x), int(cluster.pos.y) })) {
			auto clusterDesc = ClusterDescription().setId(cluster.id).setPos({ cluster.pos.x, cluster.pos.y })
				.setVel({ cluster.vel.x, cluster.vel.y })
				.setAngle(cluster.angle)
				.setAngularVel(cluster.angularVel).setMetadata(ClusterMetadata());

			for (int j = 0; j < cluster.numCells; ++j) {
				CellData const& cell = cluster.cells[j];
				auto pos = cell.absPos;
				auto id = cell.id;
				connectingCellIds.clear();
				for (int i = 0; i < cell.numConnections; ++i) {
					connectingCellIds.emplace_back(cell.connections[i]->id);
				}
				clusterDesc.addCell(
					CellDescription().setPos({ pos.x, pos.y }).setMetadata(CellMetadata())
					.setEnergy(cell.energy).setId(id).setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::COMPUTER))
					.setConnectingCells(connectingCellIds).setMaxConnections(CELL_MAX_BONDS).setFlagTokenBlocked(false)
					.setTokenBranchNumber(0).setMetadata(CellMetadata())
				);
			}
			result.addCluster(clusterDesc);
		}
	}

	for (int i = 0; i < _cudaData.numParticles; ++i) {
		ParticleData const& particle = _cudaData.particles[i];
		if (requiredRect.isContained({ int(particle.pos.x), int(particle.pos.y) })) {
			result.addParticle(ParticleDescription().setId(particle.id).setPos({ particle.pos.x, particle.pos.y })
				.setVel({ particle.vel.x, particle.vel.y }).setEnergy(particle.energy));
		}
	}

	return result;
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
	cudaCell.nextTimestep = nullptr;

	cellByIds.insert_or_assign(cudaCell.id, &cudaCell);
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

