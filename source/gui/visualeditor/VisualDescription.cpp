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

void VisualDescription::setSelection(uint64_t cellIds, uint64_t particleIds)
{
}

