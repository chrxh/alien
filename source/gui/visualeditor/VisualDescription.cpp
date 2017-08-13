#include "VisualDescription.h"

DataDescription & VisualDescription::getDataRef()
{
	return _data;
}

CellDescription & VisualDescription::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _clusterIndicesByCellIds.at(cellId);
	int cellIndex = _cellIndicesByCellIds.at(cellId);
	CellClusterDescription &clusterDesc = _data.clusters[clusterIndex].getValue();
	return clusterDesc.cells[cellIndex].getValue();
}

EnergyParticleDescription & VisualDescription::getParticleDescRef(uint64_t particleId)
{
	int particleIndex = _particleIndicesByParticleIds.at(particleId);
	return _data.particles[particleIndex].getValue();
}

bool VisualDescription::isCellPresent(uint64_t cellId)
{
	return _cellIds.find(cellId) != _cellIds.end();
}

bool VisualDescription::isParticlePresent(uint64_t particleId)
{
	return _particleIds.find(particleId) != _particleIds.end();
}

void VisualDescription::setData(DataDescription const &data)
{
	updateInternals(data);
}

void VisualDescription::setSelection(list<uint64_t> const &cellIds, list<uint64_t> const &particleIds)
{
	_selectedCellIds = set<uint64_t>(cellIds.begin(), cellIds.end());
	_selectedParticleIds = set<uint64_t>(particleIds.begin(), particleIds.end());
	_selectedClusterIds.clear();
	for (uint64_t cellId : cellIds) {
		auto clusterIdByCellIdIter = _clusterIdsByCellIds.find(cellId);
		if (clusterIdByCellIdIter != _clusterIdsByCellIds.end()) {
			_selectedClusterIds.insert(clusterIdByCellIdIter->second);
		}
	}
}

bool VisualDescription::isInSelection(list<uint64_t> const & ids) const
{
	for (uint64_t id : ids) {
		if (!isInSelection(id)) {
			return false;
		}
	}
	return true;
}

bool VisualDescription::isInSelection(uint64_t id) const
{
	return (_selectedCellIds.find(id) != _selectedCellIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
}

bool VisualDescription::isInExtendedSelection(uint64_t id) const
{
	auto clusterIdByCellIdIter = _clusterIdsByCellIds.find(id);
	if (clusterIdByCellIdIter != _clusterIdsByCellIds.end()) {
		uint64_t clusterId = clusterIdByCellIdIter->second;
		return (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
	}
	return false;
}

void VisualDescription::moveSelection(QVector2D const &delta)
{
	for (uint64_t cellId : _selectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _clusterIndicesByCellIds.at(cellId);
			int cellIndex = _cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos.setValue(cellDesc.pos.getValue() + delta);
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			EnergyParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos.setValue(particleDesc.pos.getValue() + delta);
		}
	}
}

void VisualDescription::moveExtendedSelection(QVector2D const & delta)
{
	list<uint64_t> extSelectedCellIds;
	for (auto clusterIdByCellId : _clusterIdsByCellIds) {
		uint64_t cellId = clusterIdByCellId.first;
		uint64_t clusterId = clusterIdByCellId.second;
		if (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end()) {
			extSelectedCellIds.push_back(cellId);
		}
	}

	for (uint64_t cellId : extSelectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _clusterIndicesByCellIds.at(cellId);
			int cellIndex = _cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos.setValue(cellDesc.pos.getValue() + delta);
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			EnergyParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos.setValue(particleDesc.pos.getValue() + delta);
		}
	}
}

void VisualDescription::setToUnmodified()
{
	for (auto &clusterT : _data.clusters) {
		if (clusterT.isDeleted()) { continue; }
		auto &clusterD = clusterT.getValue();
		for (auto &cellT : clusterD.cells) {
			if (cellT.isDeleted()) { continue; }
			auto &cellD = cellT.getValue();
			cellD.setAsUnmodified();
		}
	}

	for (auto &particleT : _data.particles) {
		if (particleT.isDeleted()) { continue; }
		auto &particleD = particleT.getValue();
		particleD.setAsUnmodified();
	}
}

void VisualDescription::updateAfterCellReconnections()
{
	_clusterIdsByCellIds.clear();
	_clusterIndicesByCellIds.clear();
	_cellIndicesByCellIds.clear();

	int clusterIndex = 0;
	for (auto const &cluster : getUndeletedElements(_data.clusters)) {
		int cellIndex = 0;
		for (auto const &cell : getUndeletedElements(cluster.cells)) {
			_clusterIdsByCellIds[cell.id] = cluster.id;
			_clusterIndicesByCellIds[cell.id] = clusterIndex;
			_cellIndicesByCellIds[cell.id] = cellIndex;
			++cellIndex;
		}
		++clusterIndex;
	}

	_selectedClusterIds.clear();
	for (uint64_t selectedCellId : _selectedCellIds) {
		_selectedClusterIds.insert(_clusterIdsByCellIds.at(selectedCellId));
	}
}

void VisualDescription::updateInternals(DataDescription const &data)
{
	_data = data;
	_cellIds.clear();
	_particleIds.clear();
	_clusterIdsByCellIds.clear();
	_clusterIndicesByCellIds.clear();
	_cellIndicesByCellIds.clear();
	_particleIndicesByParticleIds.clear();

	int clusterIndex = 0;
	for (auto const &cluster : getUndeletedElements(_data.clusters)) {
		int cellIndex = 0;
		for (auto const &cell : getUndeletedElements(cluster.cells)) {
			_clusterIdsByCellIds[cell.id] = cluster.id;
			_clusterIndicesByCellIds[cell.id] = clusterIndex;
			_cellIndicesByCellIds[cell.id] = cellIndex;
			_cellIds.insert(cell.id);
			++cellIndex;
		}
		++clusterIndex;
	}

	int particleIndex = 0;
	for (auto const &particle : getUndeletedElements(_data.particles)) {
		_particleIndicesByParticleIds[particle.id] = particleIndex;
		_particleIds.insert(particle.id);
		++particleIndex;
	}
}

