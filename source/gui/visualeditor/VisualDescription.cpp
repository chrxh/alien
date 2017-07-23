#include "VisualDescription.h"

DataDescription & VisualDescription::getDataRef()
{
	return _data;
}

map<uint64_t, CellDescription> VisualDescription::getCellDescsByIds() const
{
	map<uint64_t, CellDescription> result;
	for (auto const &clusterT : _data.clusters) {
		auto const &clusterD = clusterT.getValue();
		for (auto const &cellT : clusterD.cells) {
			auto const &cellD = cellT.getValue();
			result[cellD.id] = cellD;
		}
	}
	return result;
}

void VisualDescription::setData(DataDescription const & data)
{
	_data = data;
}

void VisualDescription::setSelection(set<uint64_t> const &cellIds, set<uint64_t> const &particleIds)
{
	_selectedCellIds = cellIds;
	_selectedParticleIds = particleIds;
}

bool VisualDescription::isInSelection(uint64_t id) const
{
	return (_selectedCellIds.find(id) != _selectedCellIds.end() || _selectedParticleIds.find(id) != _selectedParticleIds.end());
}

bool VisualDescription::isInExtendedSelection(uint64_t id) const
{
	return false;
}

