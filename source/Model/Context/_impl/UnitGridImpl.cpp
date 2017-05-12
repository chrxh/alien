#include "model/Context/Unit.h"
#include "model/Context/SpaceMetric.h"

#include "UnitGridImpl.h"

UnitGridImpl::UnitGridImpl(QObject * parent)
	: UnitGrid(parent)
{
}

UnitGridImpl::~UnitGridImpl()
{
}

void UnitGridImpl::init(IntVector2D gridSize, SpaceMetric* metric)
{
	if ((metric->getSize().x % gridSize.x != 0) || (metric->getSize().y % gridSize.y != 0)) {
		throw std::exception("Universe size is not a multiple of grid size.");
	}
	_metric = metric;
	_gridSize = gridSize;
	for (int x = 0; x < gridSize.x; ++x) {
		_units.push_back(vector<Unit*>(gridSize.y, nullptr));
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

Unit * UnitGridImpl::getUnitOfGridPos(IntVector2D gridPos) const
{
	return _units[gridPos.x][gridPos.y];
}

Unit * UnitGridImpl::getUnitOfMapPos(QVector2D pos) const
{
	return getUnitOfGridPos(getGridPosOfMapPos(pos));
}

IntVector2D UnitGridImpl::getGridPosOfMapPos(QVector2D pos) const
{
	IntVector2D intPos = pos;// _metric->correctPositionWithIntPrecision(pos);
	auto size = _metric->getSize();
	intPos.restrictToRect({ { 0, 0 }, { size.x - 1, size.y - 1 } });
	IntVector2D compartmentSize = calcCompartmentSize();
	return { intPos.x / compartmentSize.x, intPos.y / compartmentSize.y };
}

IntRect UnitGridImpl::calcCompartmentRect(IntVector2D gridPos) const
{
	IntVector2D universeSize = _metric->getSize();
	IntVector2D p1 = { universeSize.x * gridPos.x / _gridSize.x, universeSize.y * gridPos.y / _gridSize.y };
	IntVector2D p2 = { universeSize.x * (gridPos.x + 1) / _gridSize.x - 1, universeSize.y * (gridPos.y + 1) / _gridSize.y - 1 };
	return{ p1, p2 };
}

IntVector2D UnitGridImpl::calcCompartmentSize() const
{
	IntVector2D universeSize = _metric->getSize();
	return{ universeSize.x / _gridSize.x, universeSize.y / _gridSize.y };
}

