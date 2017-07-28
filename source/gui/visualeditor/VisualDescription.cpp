#include "VisualDescription.h"

DataDescription & VisualDescription::getDataRef()
{
	return _data;
}

map<uint64_t, CellDescription> const& VisualDescription::getCellDescsByIds() const
{
	return _cellDescsByIds;
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

}

void VisualDescription::updateInternals()
{
	_clusterIdsByCellIds.clear();
	_cellDescsByIds.clear();
	for (auto const &clusterT : _data.clusters) {
		auto const &cluster = clusterT.getValue();
		for (auto const &cellT : cluster.cells) {
			auto const &cell = cellT.getValue();
			_clusterIdsByCellIds[cell.id] = cluster.id;
			_cellDescsByIds[cell.id] = cell;
		}
	}
}

