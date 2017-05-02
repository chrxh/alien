#include "model/context/unit.h"
#include "model/context/spacemetric.h"

#include "gridimpl.h"

GridImpl::GridImpl(QObject * parent)
	: Grid(parent)
{
}

GridImpl::~GridImpl()
{
	deleteUnits();
}

void GridImpl::init(IntVector2D gridSize, SpaceMetric* metric)
{
	deleteUnits();

	if (_metric != metric) {
		delete _metric;
		_metric = metric;
	}
	_gridSize = gridSize;
	for (int x = 0; x < gridSize.x; ++x) {
		_units.push_back(std::vector<Unit*>(gridSize.y, nullptr));
	}
}

void GridImpl::registerUnit(IntVector2D gridPos, Unit * unit)
{
	_units[gridPos.x][gridPos.y] = unit;
}

IntVector2D GridImpl::getSize() const
{
	return _gridSize;
}

Unit * GridImpl::getUnit(IntVector2D gridPos) const
{
	return _units[gridPos.x][gridPos.y];
}

IntRect GridImpl::calcMapRect(IntVector2D gridPos) const
{
	IntVector2D universeSize = _metric->getSize();
	IntVector2D p1 = { universeSize.x * gridPos.x / _gridSize.x, universeSize.y * gridPos.y / _gridSize.y };
	IntVector2D p2 = { universeSize.x * (gridPos.x + 1) / _gridSize.x - 1, universeSize.y * (gridPos.y + 1) / _gridSize.y - 1 };
	return{ p1, p2 };
}

void GridImpl::deleteUnits()
{
	for (auto const& unitVec : _units) {
		for (auto const& unit : unitVec) {
			delete unit;
		}
	}
}
