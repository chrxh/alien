#include "model/context/Unit.h"
#include "model/context/SpaceMetric.h"

#include "UnitGridImpl.h"

UnitGridImpl::UnitGridImpl(QObject * parent)
	: UnitGrid(parent)
{
}

void UnitGridImpl::init(IntVector2D gridSize, SpaceMetric* metric)
{
	if ((metric->getSize().x % gridSize.x != 0) || (metric->getSize().y % gridSize.y != 0)) {
		throw std::exception("Universe size is not a multiple of grid size.");
	}

	if (_metric != metric) {
		delete _metric;
		_metric = metric;
	}
	_gridSize = gridSize;
	for (int x = 0; x < gridSize.x; ++x) {
		_units.push_back(std::vector<Unit*>(gridSize.y, nullptr));
	}
}

void UnitGridImpl::registerUnit(IntVector2D gridPos, Unit * unit)
{
	_units[gridPos.x][gridPos.y] = unit;
}

IntVector2D UnitGridImpl::getSize() const
{
	return _gridSize;
}

Unit * UnitGridImpl::getUnit(IntVector2D gridPos) const
{
	return _units[gridPos.x][gridPos.y];
}

IntRect UnitGridImpl::calcCompartmentRect(IntVector2D gridPos) const
{
	IntVector2D universeSize = _metric->getSize();
	IntVector2D p1 = { universeSize.x * gridPos.x / _gridSize.x, universeSize.y * gridPos.y / _gridSize.y };
	IntVector2D p2 = { universeSize.x * (gridPos.x + 1) / _gridSize.x - 1, universeSize.y * (gridPos.y + 1) / _gridSize.y - 1 };
	return{ p1, p2 };
}

