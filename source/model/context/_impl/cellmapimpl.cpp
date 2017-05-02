#include "model/context/SpaceMetric.h"
#include "model/context/UnitContext.h"
#include "model/context/MapCompartment.h"
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
	if (_compartment->isPointInCompartment(intPos)) {
		_compartment->convertAbsToRelPosition(intPos);
		_cellGrid[intPos.x][intPos.y] = cell;
	}
	else {
		auto context = _compartment->getNeighborContext(intPos);
		_compartment->convertAbsToRelPosition(intPos);
		static_cast<CellMapImpl*>(context->getCellMap())->_cellGrid[intPos.x][intPos.y] = cell;
	}
}

void CellMapImpl::removeCellIfPresent(QVector3D pos, Cell * cell)
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	IntVector2D intPosM = _metric->shiftPosition(intPos, { -1, -1 });
	IntVector2D intPosP = _metric->shiftPosition(intPos, { 1, 1 });

	if (_compartment->isPointInCompartment(intPosM) && _compartment->isPointInCompartment(intPosP)) {

		auto removeCellIfPresent = [&](int const &x, int const &y, Cell* cell) {
			if (_cellGrid[x][y] == cell)
				_cellGrid[x][y] = nullptr;
		};

		removeCellIfPresent(intPosM.x, intPosM.y, cell);
		removeCellIfPresent(intPos.x, intPosM.y, cell);
		removeCellIfPresent(intPosP.x, intPosM.y, cell);

		removeCellIfPresent(intPosM.x, intPos.y, cell);
		removeCellIfPresent(intPos.x, intPos.y, cell);
		removeCellIfPresent(intPosP.x, intPos.y, cell);

		removeCellIfPresent(intPosM.x, intPosP.y, cell);
		removeCellIfPresent(intPos.x, intPosP.y, cell);
		removeCellIfPresent(intPosP.x, intPosP.y, cell);
	}
}

Cell* CellMapImpl::getCell(QVector3D pos) const
{
	IntVector2D intPos = _metric->correctPositionWithIntPrecision(pos);
	if (_compartment->isPointInCompartment(intPos)) {
		_compartment->convertAbsToRelPosition(intPos);
		return _cellGrid[intPos.x][intPos.y];
	}
	else {
		auto context = _compartment->getNeighborContext(intPos);
		_compartment->convertAbsToRelPosition(intPos);
		return static_cast<CellMapImpl*>(context->getCellMap())->_cellGrid[intPos.x][intPos.y];
	}
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
				Cell* cell = getCellFast(_metric->shiftPosition(intPos, { rx, ry }));
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
				Cell* cell = getCellFast(_metric->shiftPosition(intPos, { rx, ry }));
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
