#include <functional>

#include "SpacePropertiesImpl.h"
#include "UnitContext.h"
#include "ModelInterface/Settings.h"
#include "Cluster.h"

#include "Cell.h"
#include "CellMapImpl.h"

CellMapImpl::CellMapImpl(QObject* parent /*= nullptr*/)
	: CellMap(parent)
{
}

CellMapImpl::~CellMapImpl()
{
	deleteCellMap();
}

void CellMapImpl::init(SpacePropertiesImpl* metric, MapCompartment* compartment)
{
	_metric = metric;
	_compartment = compartment;

	deleteCellMap();
	_size = _compartment->getSize();
	_cellGrid = new Cell**[_size.x];
	for (int x = 0; x < _size.x; ++x) {
		_cellGrid[x] = new Cell*[_size.y];
	}
	clear();
}

void CellMapImpl::clear()
{
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y)
			_cellGrid[x][y] = nullptr;
}

void CellMapImpl::setCell(QVector2D pos, Cell * cell)
{
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	locateCell(intPos) = cell;
}

void CellMapImpl::removeCellIfPresent(QVector2D pos, Cell * cellToRemove)
{
/*
	//TEMP
	IntVector2D intPosC = _metric->correctPositionWithIntPrecision(pos);
	_compartment->convertAbsToRelPosition(intPosC);
	Cell*** cellGrid = _cellGrid;
	if (!_compartment->isPointInCompartment(intPosC)) {
		cellGrid = static_cast<CellMapImpl*>(_compartment->getNeighborContext(intPosC)->getCellMap())->_cellGrid;
	}
	if (_cellGrid[intPosC.x][intPosC.y] == cellToRemove) {
		_cellGrid[intPosC.x][intPosC.y] = nullptr;
	}
*/

	IntVector2D intPosC = _metric->correctPositionAndConvertToIntVector(pos);
	IntVector2D intPosM = _metric->shiftPosition(intPosC, { -1, -1 });
	IntVector2D intPosP = _metric->shiftPosition(intPosC, { +1, +1 });

	std::function<void(IntVector2D &&, Cell*)> removeCellIfPresent;
	if (_compartment->isPointInCompartment(intPosM) && _compartment->isPointInCompartment(intPosP)) {
//	if (_compartment->isPointInCompartment(intPosC)) {
		removeCellIfPresent = [&](IntVector2D && intPos, Cell* cell) {
			if (_cellGrid[intPos.x][intPos.y] == cell) {
				_cellGrid[intPos.x][intPos.y] = nullptr;
			}
		};
		_compartment->convertAbsToRelPosition(intPosC);
		_compartment->convertAbsToRelPosition(intPosM);
		_compartment->convertAbsToRelPosition(intPosP);
	}
	else {
		removeCellIfPresent = [&](IntVector2D && intPos, Cell* cell) {
			Cell*** cellGrid = _cellGrid;
			if (!_compartment->isPointInCompartment(intPos)) {
				cellGrid = static_cast<CellMapImpl*>(_compartment->getNeighborContext(intPos)->getCellMap())->_cellGrid;
			}
			_compartment->convertAbsToRelPosition(intPos);
			if (cellGrid[intPos.x][intPos.y] == cell) {
				cellGrid[intPos.x][intPos.y] = nullptr;
			}
		};
	}

	removeCellIfPresent({ intPosM.x, intPosM.y }, cellToRemove);
	removeCellIfPresent({ intPosC.x, intPosM.y }, cellToRemove);
	removeCellIfPresent({ intPosP.x, intPosM.y }, cellToRemove);

	removeCellIfPresent({ intPosM.x, intPosC.y }, cellToRemove);
	removeCellIfPresent({ intPosC.x, intPosC.y }, cellToRemove);
	removeCellIfPresent({ intPosP.x, intPosC.y }, cellToRemove);

	removeCellIfPresent({ intPosM.x, intPosP.y }, cellToRemove);
	removeCellIfPresent({ intPosC.x, intPosP.y }, cellToRemove);
	removeCellIfPresent({ intPosP.x, intPosP.y }, cellToRemove);
}

Cell* CellMapImpl::getCell(QVector2D pos) const
{
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	return locateCell(intPos);
}

CellClusterSet CellMapImpl::getNearbyClusters(QVector2D const& pos, qreal r) const
{
	CellClusterSet clusters;
	int rc = qCeil(r);
	for (int rx = pos.x() - rc; rx < pos.x() + rc + 1; ++rx)
		for (int ry = pos.y() - rc; ry < pos.y() + rc + 1; ++ry) {
			if (QVector2D(static_cast<qreal>(rx) - pos.x(), static_cast<qreal>(ry) - pos.y()).length() < r + Const::AlienPrecision) {
				Cell* cell = getCell(QVector2D(rx, ry));
				if (cell) {
					clusters.insert(cell->getCluster());
				}
			}
		}
	return clusters;
}

Cluster * CellMapImpl::getNearbyClusterFast(const QVector2D & pos, qreal r, qreal minMass, qreal maxMass, Cluster * exclude) const
{
	int step = qCeil(qSqrt(minMass + Const::AlienPrecision)) + 3;  //horizontal or vertical length of cell cluster >= minDim
	int rc = qCeil(r);
	qreal rs = r*r + Const::AlienPrecision;

	//grid scan
	Cluster* closestCluster = 0;
	qreal closestClusterDist = 0.0;

	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	for (int rx = -rc; rx <= rc; rx += step)
		for (int ry = -rc; ry <= rc; ry += step) {
			if (static_cast<qreal>(rx*rx + ry*ry) < rs) {
				Cell*& cell = locateCell(_metric->shiftPosition(intPos, { rx, ry }));
				if (cell) {
					Cluster* cluster = cell->getCluster();
					if (cluster != exclude) {

						//compare masses
						qreal mass = cluster->getMass();
						if (mass >= (minMass - Const::AlienPrecision) && mass <= (maxMass + Const::AlienPrecision)) {

							//calc and compare dist
							qreal dist = _metric->displacement(cell->calcPosition(), pos).length();
							if (!closestCluster || (dist < closestClusterDist)) {
								closestCluster = cluster;
								closestClusterDist = dist;
							}
						}
					}
				}
			}
		}
	return closestCluster;
}

QList<Cell*> CellMapImpl::getNearbySpecificCells(const QVector2D & pos, qreal r, CellSelectFunction selection) const
{
	QList< Cell* > cells;
	int rCeil = qCeil(r);
	qreal rs = r*r + Const::AlienPrecision;
	IntVector2D intPos = _metric->correctPositionAndConvertToIntVector(pos);
	for (int rx = -rCeil; rx <= rCeil; ++rx)
		for (int ry = -rCeil; ry <= rCeil; ++ry)
			if (static_cast<qreal>(rx*rx + ry*ry) < rs) {
				Cell*& cell = locateCell(_metric->shiftPosition(intPos, { rx, ry }));
				if (cell) {
					if (selection(cell)) {
						cells << cell;
					}
				}
			}
	return cells;
}

void CellMapImpl::deleteCellMap()
{
	if (_cellGrid != nullptr) {
		for (int x = 0; x < _size.x; ++x) {
			delete[] _cellGrid[x];
		}
		delete[] _cellGrid;
	}
}
