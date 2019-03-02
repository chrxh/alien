#include "Base/NumberGenerator.h"
#include "ModelBasic/Descriptions.h"
#include "ModelBasic/ChangeDescriptions.h"
#include "ModelBasic/Physics.h"

#include "DataConverter.h"

DataConverter::DataConverter(SimulationDataForAccess& cudaData, NumberGenerator* numberGen)
	: _cudaData(cudaData), _numberGen(numberGen)
{}

void DataConverter::updateData(DataChangeDescription const & data)
{
	for (auto const& cluster : data.clusters) {
		if (cluster.isDeleted()) {
			markDelCluster(cluster.getValue().id);
		}
		if (cluster.isModified()) {
			markModifyCluster(cluster.getValue());
		}
	}
	for (auto const& particle : data.particles) {
		if (particle.isDeleted()) {
			markDelParticle(particle.getValue().id);
		}
		if (particle.isModified()) {
			markModifyParticle(particle.getValue());
		}
	}
	processDeletionsAndModifications();

	for (auto const& cluster : data.clusters) {
		if (cluster.isAdded()) {
			addCluster(cluster.getValue());
		}
	}
	for (auto const& particle : data.particles) {
		if (particle.isAdded()) {
			addParticle(particle.getValue());
		}
	}
}

void DataConverter::addCluster(ClusterDescription const& clusterDesc)
{
	if (!clusterDesc.cells) {
		return;
	}

	ClusterData& cudaCluster = _cudaData.clusters[_cudaData.numClusters++];
	cudaCluster.id = clusterDesc.id == 0 ? _numberGen->getId() : clusterDesc.id;
	QVector2D clusterPos = clusterDesc.pos ? clusterPos = *clusterDesc.pos : clusterPos = clusterDesc.getClusterPosFromCells();
	cudaCluster.pos = { clusterPos.x(), clusterPos.y() };
	cudaCluster.vel = { clusterDesc.vel->x(), clusterDesc.vel->y() };
	cudaCluster.angle = *clusterDesc.angle;
	cudaCluster.angularVel = *clusterDesc.angularVel;
	cudaCluster.numCells = clusterDesc.cells ? clusterDesc.cells->size() : 0;
	cudaCluster.decompositionRequired = false;
	cudaCluster.locked = 0;
	cudaCluster.clusterToFuse = nullptr;
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

void DataConverter::addParticle(ParticleDescription const & particleDesc)
{
	ParticleData& cudaParticle = _cudaData.particles[_cudaData.numParticles++];
	cudaParticle.id = cudaParticle.id == 0 ? _numberGen->getId() : particleDesc.id;
	cudaParticle.pos = { particleDesc.pos->x(), particleDesc.pos->y() };
	cudaParticle.vel = { particleDesc.vel->x(), particleDesc.vel->y() };
	cudaParticle.energy = *particleDesc.energy;
	cudaParticle.locked = 0;
	cudaParticle.alive = true;
}

void DataConverter::markDelCluster(uint64_t clusterId)
{
	_clusterIdsToDelete.insert(clusterId);
}

void DataConverter::markDelParticle(uint64_t particleId)
{
	_particleIdsToDelete.insert(particleId);
}

void DataConverter::markModifyCluster(ClusterChangeDescription const & clusterDesc)
{
	_clusterToModifyById.insert_or_assign(clusterDesc.id, clusterDesc);
	for (auto const& cellTracker : clusterDesc.cells) {
		if (cellTracker.isModified()) {
			_cellToModifyById.insert_or_assign(cellTracker->id, cellTracker.getValue());
		}
	}
}

void DataConverter::markModifyParticle(ParticleChangeDescription const & particleDesc)
{
	_particleToModifyById.insert_or_assign(particleDesc.id, particleDesc);
}

void DataConverter::processDeletionsAndModifications()
{
	if (_clusterIdsToDelete.empty() && _particleIdsToDelete.empty() && _clusterToModifyById.empty()
		&& _cellToModifyById.empty() && _particleToModifyById.empty()) {
		return;
	}

	//delete and modify clusters
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

		if (_clusterToModifyById.find(clusterId) != _clusterToModifyById.end()) {
			applyChangeDescription(cluster, _clusterToModifyById.at(clusterId));
		}
	}
	_cudaData.numClusters -= clusterIndexCopyOffset;

	//delete and modify cells
	int cellIndexCopyOffset = 0;
	std::unordered_map<CellData*, CellData*> newByOldCellData;
	for (int index = 0; index < _cudaData.numCells; ++index) {
		CellData& cell = _cudaData.cells[index];
		uint64_t cellId = cell.id;
		if (cellIdsToDelete.find(cellId) != cellIdsToDelete.end()) {
			++cellIndexCopyOffset;
		}
		else if (cellIndexCopyOffset > 0) {
			newByOldCellData.insert_or_assign(&cell, &_cudaData.cells[index - cellIndexCopyOffset]);
			_cudaData.cells[index - cellIndexCopyOffset] = cell;
		}

		if (_cellToModifyById.find(cellId) != _cellToModifyById.end()) {
			uint64_t clusterId = cell.cluster->id;
			applyChangeDescription(cell, _cellToModifyById.at(cellId), _clusterToModifyById.at(clusterId));
		}
	}
	_cudaData.numCells -= cellIndexCopyOffset;

	//delete and modify particles
	int particleIndexCopyOffset = 0;
	for (int index = 0; index < _cudaData.numParticles; ++index) {
		ParticleData& particle = _cudaData.particles[index];
		uint64_t particleId = particle.id;
		if (_particleIdsToDelete.find(particleId) != _particleIdsToDelete.end()) {
			++particleIndexCopyOffset;
		}
		else if (particleIndexCopyOffset > 0) {
			_cudaData.particles[index - particleIndexCopyOffset] = particle;
		}

		if (_particleToModifyById.find(particleId) != _particleToModifyById.end()) {
			applyChangeDescription(particle, _particleToModifyById.at(particleId));
		}
	}
	_cudaData.numParticles -= particleIndexCopyOffset;

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
					.setConnectingCells(connectingCellIds).setMaxConnections(cell.maxConnections).setFlagTokenBlocked(false)
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
	cudaCell.id = cellDesc.id == 0 ? _numberGen->getId() : cellDesc.id;
	cudaCell.cluster = &cudaCluster;
	cudaCell.absPos = { cellDesc.pos->x(), cellDesc.pos->y() };
	QVector2D relPos = cellDesc.getPosRelativeTo(cluster);
	cudaCell.relPos = { relPos.x(), relPos.y() };
	cudaCell.energy = *cellDesc.energy;
	cudaCell.maxConnections = *cellDesc.maxConnections;
	if (cellDesc.connectingCells) {
		cudaCell.numConnections = cellDesc.connectingCells->size();
	}
	else {
		cudaCell.numConnections = 0;
	}
	cudaCell.protectionCounter = 0;
	cudaCell.alive = true;
	cudaCell.nextTimestep = nullptr;
	auto vel = Physics::tangentialVelocity(*cellDesc.pos - *cluster.pos, { *cluster.vel, *cluster.angularVel });
	cudaCell.vel = { vel.x(), vel.y() };

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

namespace
{
	void convert(QVector2D const& input, float2& output)
	{
		output.x = input.x();
		output.y = input.y();
	}
}

void DataConverter::applyChangeDescription(ParticleData & particle, ParticleChangeDescription const & particleChanges)
{
	if (particleChanges.pos) {
		QVector2D newPos = particleChanges.pos.getValue();
		convert(newPos, particle.pos);
	}
	if (particleChanges.vel) {
		QVector2D newVel = particleChanges.vel.getValue();
		convert(newVel, particle.vel);
	}
	if (particleChanges.energy) {
		particle.energy = particleChanges.energy.getValue();
	}
}

void DataConverter::applyChangeDescription(ClusterData & cluster, ClusterChangeDescription const & clusterChanges)
{
	if (clusterChanges.pos) {
		QVector2D newPos = clusterChanges.pos.getValue();
		convert(newPos, cluster.pos);
	}
	if (clusterChanges.vel) {
		QVector2D newVel = clusterChanges.vel.getValue();
		convert(newVel, cluster.vel);
	}
	if (clusterChanges.angle) {
		cluster.angle = clusterChanges.angle.getValue();
	}
	if (clusterChanges.angularVel) {
		cluster.angularVel = clusterChanges.angularVel.getValue();
	}
	updateCellVelocities(cluster);
	updateAngularMass(cluster);
}

void DataConverter::applyChangeDescription(CellData & cell, CellChangeDescription const & cellChanges
	, ClusterChangeDescription const& clusterChanges)
{
	if (cellChanges.pos) {
		QVector2D newAbsPos = cellChanges.pos.getValue();
		convert(newAbsPos, cell.absPos);

		CellDescription newCell(cellChanges);
		ClusterDescription newCluster(clusterChanges);
		QVector2D newRelPos = newCell.getPosRelativeTo(newCluster);
		convert(newRelPos, cell.relPos);
	}
	if (cellChanges.energy) {
		cell.energy = cellChanges.energy.getValue();
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

void DataConverter::updateCellVelocities(ClusterData & cluster)
{
	for (int i = 0; i < cluster.numCells; ++i) {
		auto& cudaCell = cluster.cells[i];
		QVector2D clusterPos(cluster.pos.x, cluster.pos.y);
		QVector2D clusterVel(cluster.vel.x, cluster.vel.y);
		QVector2D cellPos(cudaCell.absPos.x, cudaCell.absPos.y);
		auto vel = Physics::tangentialVelocity(cellPos - clusterPos, { clusterVel, cluster.angularVel });
		cudaCell.vel = { vel.x(), vel.y() };
	}
}

