#include "model/context/SpaceMetric.h"
#include "model/context/UnitContext.h"
#include "model/ModelSettings.h"
#include "model/entities/Cell.h"
#include "model/entities/CellCluster.h"

#include "CellMapImpl.h"

CellMapImpl::CellMapImpl(QObject* parent /*= nullptr*/)
	: CellMap(parent)
{
}

CellMapImpl::~CellMapImpl()
{
	deleteCellMap();
}

void CellMapImpl::init(SpaceMetric* metric, MapCompartment* compartment)
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

void CellMapImpl::setCell(QVector3D pos, Cell * cell)
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	locateCell(intPos) = cell;
}

void CellMapImpl::removeCellIfPresent(QVector3D pos, Cell * cellToRemove)
{
	IntVector2D intPosC = _metric->correctPositionWithIntPrecision(pos);
	IntVector2D intPosM = _metric->shiftPosition(intPosC, { -1, -1 });
	IntVector2D intPosP = _metric->shiftPosition(intPosC, { +1, +1 });

	auto removeCellIfPresent = [&](IntVector2D && intPos, Cell* cell) {
		if (_cellGrid[intPos.x][intPos.y] == cell) {
			_cellGrid[intPos.x][intPos.y] = nullptr;
		}
	};

	if (_compartment->isPointInCompartment(intPosM) && _compartment->isPointInCompartment(intPosP)) {
		intPosC = _compartment->convertAbsToRelPosition(intPosC);
		intPosM = _compartment->convertAbsToRelPosition(intPosM);
		intPosP = _compartment->convertAbsToRelPosition(intPosP);
	}
	else {
		auto removeCellIfPresent = [&](IntVector2D && intPos, Cell* cell) {
			auto cellMap = static_cast<CellMapImpl*>(_compartment->getNeighborContext(intPos)->getCellMap());
			intPos = _compartment->convertAbsToRelPosition(intPos);
			if (cellMap->_cellGrid[intPos.x][intPos.y] == cell) {
				cellMap->_cellGrid[intPos.x][intPos.y] = nullptr;
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

Cell* CellMapImpl::getCell(QVector3D pos) const
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	return locateCell(intPos);
}

CellClusterSet CellMapImpl::getNearbyClusters(QVector3D const& pos, qreal r) const
{
	CellClusterSet clusters;
	int rc = qCeil(r);
	for (int rx = pos.x() - rc; rx < pos.x() + rc + 1; ++rx)
		for (int ry = pos.y() - rc; ry < pos.y() + rc + 1; ++ry) {
			if (QVector3D(static_cast<qreal>(rx) - pos.x(), static_cast<qreal>(ry) - pos.y(), 0).length() < r + ALIEN_PRECISION) {
				Cell* cell = getCell(QVector3D(rx, ry, 0));
				if (cell) {
					clusters.insert(cell->getCluster());
				}
			}
		}
	return clusters;
}

CellCluster * CellMapImpl::getNearbyClusterFast(const QVector3D & pos, qreal r, qreal minMass, qreal maxMass, CellCluster * exclude) const
{
	int step = qCeil(qSqrt(minMass + ALIEN_PRECISION)) + 3;  //horizontal or vertical length of cell cluster >= minDim
	int rc = qCeil(r);
	qreal rs = r*r + ALIEN_PRECISION;

	//grid scan
	CellCluster* closestCluster = 0;
	qreal closestClusterDist = 0.0;

	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	for (int rx = -rc; rx <= rc; rx += step)
		for (int ry = -rc; ry <= rc; ry += step) {
			if (static_cast<qreal>(rx*rx + ry*ry) < rs) {
				Cell*& cell = locateCell(_metric->shiftPosition(intPos, { rx, ry }));
				if (cell) {
					CellCluster* cluster = cell->getCluster();
					if (cluster != exclude) {

						//compare masses
						qreal mass = cluster->getMass();
						if (mass >= (minMass - ALIEN_PRECISION) && mass <= (maxMass + ALIEN_PRECISION)) {

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

QList<Cell*> CellMapImpl::getNearbySpecificCells(const QVector3D & pos, qreal r, CellSelectFunction selection) const
{
	QList< Cell* > cells;
	int rCeil = qCeil(r);
	qreal rs = r*r + ALIEN_PRECISION;
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
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

void CellMapImpl::serializePrimitives(QDataStream & stream) const
{
	//determine number of cell entries
	quint32 numEntries = 0;
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y)
			if (_cellGrid[x][y])
				numEntries++;
	stream << numEntries;

	//write cell entries
	for (int x = 0; x < _size.x; ++x)
		for (int y = 0; y < _size.y; ++y) {
			Cell* cell = _cellGrid[x][y];
			if (cell) {
				stream << x << y << cell->getId();
			}

		}
}

void CellMapImpl::deserializePrimitives(QDataStream & stream, const QMap<quint64, Cell*>& oldIdCellMap)
{
	quint32 numEntries = 0;
	qint32 x = 0;
	qint32 y = 0;
	quint64 oldId = 0;
	stream >> numEntries;
	for (quint32 i = 0; i < numEntries; ++i) {
		stream >> x >> y >> oldId;
		_cellGrid[x][y] = oldIdCellMap[oldId];
	}
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
