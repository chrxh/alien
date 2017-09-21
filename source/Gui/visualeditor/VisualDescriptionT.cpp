#include "VisualDescriptionT.h"

DataDescription & VisualDescription::getDataRef()
{
	return _data;
}

CellDescription & VisualDescription::getCellDescRef(uint64_t cellId)
{
	int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
	ClusterDescription &clusterDesc = _data.clusters->at(clusterIndex);
	int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
	return clusterDesc.cells->at(cellIndex);
}

ParticleDescription & VisualDescription::getParticleDescRef(uint64_t particleId)
{
	int particleIndex = _navi.particleIndicesByParticleIds.at(particleId);
	return _data.particles->at(particleIndex);
}

bool VisualDescription::isCellPresent(uint64_t cellId)
{
	return _navi.cellIds.find(cellId) != _navi.cellIds.end();
}

bool VisualDescription::isParticlePresent(uint64_t particleId)
{
	return _navi.particleIds.find(particleId) != _navi.particleIds.end();
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
		auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(cellId);
		if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
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
	auto clusterIdByCellIdIter = _navi.clusterIdsByCellIds.find(id);
	if (clusterIdByCellIdIter != _navi.clusterIdsByCellIds.end()) {
		uint64_t clusterId = clusterIdByCellIdIter->second;
		return (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
	}
	return false;
}

bool VisualDescription::areEntitiesSelected() const
{
	return !_selectedCellIds.empty() || !_selectedParticleIds.empty();
}

list<uint64_t> VisualDescription::getSelectedCellIds() const
{
	list<uint64_t> result(_selectedCellIds.begin(), _selectedCellIds.end());
	return result;
}

void VisualDescription::moveSelection(QVector2D const &delta)
{
	for (uint64_t cellId : _selectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
			int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos = *cellDesc.pos + delta;
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			ParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos = *particleDesc.pos + delta;
		}
	}
}

void VisualDescription::moveExtendedSelection(QVector2D const & delta)
{
	list<uint64_t> extSelectedCellIds;
	for (auto clusterIdByCellId : _navi.clusterIdsByCellIds) {
		uint64_t cellId = clusterIdByCellId.first;
		uint64_t clusterId = clusterIdByCellId.second;
		if (_selectedClusterIds.find(clusterId) != _selectedClusterIds.end()) {
			extSelectedCellIds.push_back(cellId);
		}
	}

	for (uint64_t cellId : extSelectedCellIds) {
		if (isCellPresent(cellId)) {
			int clusterIndex = _navi.clusterIndicesByCellIds.at(cellId);
			int cellIndex = _navi.cellIndicesByCellIds.at(cellId);
			CellDescription &cellDesc = getCellDescRef(cellId);
			cellDesc.pos = *cellDesc.pos + delta;
		}
	}

	for (uint64_t particleId : _selectedParticleIds) {
		if (isParticlePresent(particleId)) {
			ParticleDescription &particleDesc = getParticleDescRef(particleId);
			particleDesc.pos = *particleDesc.pos + delta;
		}
	}
}

void VisualDescription::updateAfterCellReconnections()
{
	_navi.update(_data);

	_selectedClusterIds.clear();
	for (uint64_t selectedCellId : _selectedCellIds) {
		if (_navi.clusterIdsByCellIds.find(selectedCellId) != _navi.clusterIdsByCellIds.end()) {
			_selectedClusterIds.insert(_navi.clusterIdsByCellIds.at(selectedCellId));
		}
	}
}

void VisualDescription::updateInternals(DataDescription const &data)
{
	_data = data;
	_navi.update(data);
}

