#include "DescriptionManager.h"

DataDescription & DescriptionManager::getDataRef()
{
	return _data;
}

void DescriptionManager::setData(DataDescription const & data)
{
	_data = data;
}

void DescriptionManager::changeCellDescription(uint64_t clusterId, CellDescription const & cell)
{
}

CellDescription DescriptionManager::getCellDescription(uint64_t cellId) const
{
	return CellDescription();
}
