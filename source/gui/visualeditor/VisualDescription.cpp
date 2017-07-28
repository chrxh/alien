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

void VisualDescription::setData(DataDescription const & data)
{
	_data = data;
	updateInternals();
}

void VisualDescription::setSelection(set<uint64_t> const &cellIds, set<uint64_t> const &particleIds)
{
	_selectedCellIds = cellIds;
	_selectedParticleIds = particleIds;
	_selectedClusterIds.clear();
	for (uint64_t cellId : cellIds) {
		auto clusterIdByCellIdIter = _clusterIdsByCellIds.find(cellId);
		if (clusterIdByCellIdIter != _clusterIdsByCellIds.end()) {
			_selectedClusterIds.insert(clusterIdByCellIdIter->second);
		}
	}
}

bool VisualDescription::isSomethingSelected()
{
	return !_selectedCellIds.empty() || !_selectedParticleIds.empty();
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
		int clusterIndex = _clusterIndicesByCellIds.at(cellId);
		int cellIndex = _cellIndicesByCellIds.at(cellId);
		CellClusterDescription &clusterDesc = _data.clusters[clusterIndex].getValue();
		CellDescription &cellDesc = getCellDescRef(cellId);
		cellDesc.pos.setValue(cellDesc.pos.getValue() + delta);
	}

	for (uint64_t particleId : _selectedParticleIds) {
		EnergyParticleDescription &particleDesc = getParticleDescRef(particleId);
		particleDesc.pos.setValue(particleDesc.pos.getValue() + delta);
	}
}

void VisualDescription::updateInternals()
{
	_clusterIdsByCellIds.clear();
	_clusterIndicesByCellIds.clear();
	_cellIndicesByCellIds.clear();
	_particleIndicesByParticleIds.clear();

	int clusterIndex = 0;
	for (auto const &clusterT : _data.clusters) {
		auto const &clusterD = clusterT.getValue();
		int cellIndex = 0;
		for (auto const &cellT : clusterD.cells) {
			auto const &cellD = cellT.getValue();
			_clusterIdsByCellIds[cellD.id] = clusterD.id;
			_clusterIndicesByCellIds[cellD.id] = clusterIndex;
			_cellIndicesByCellIds[cellD.id] = cellIndex;
			++cellIndex;
		}
		++clusterIndex;
	}

	int particleIndex = 0;
	for (auto const &particleT : _data.particles) {
		auto const &particleD = particleT.getValue();
		_particleIndicesByParticleIds[particleD.id] = particleIndex;
		++particleIndex;
	}
}

