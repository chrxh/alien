#ifndef CELLMAP_H
#define CELLMAP_H

#include "model/Definitions.h"

class CellMap
	: public QObject
{
	Q_OBJECT
public:
	CellMap(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CellMap() = default;

	virtual void init(SpaceMetric* topo, MapCompartment* compartment) = 0;
	virtual void clear() = 0;

	virtual void setCell(QVector2D pos, Cell* cell) = 0;
	virtual void removeCellIfPresent(QVector2D pos, Cell* cellToRemove) = 0;
	virtual Cell* getCell(QVector2D pos) const = 0;
	inline Cell* getCellFast(IntVector2D const& intPos) const;

	//advanced functions
	virtual CellClusterSet getNearbyClusters(QVector2D const& pos, qreal r) const = 0;
	virtual CellCluster* getNearbyClusterFast(QVector2D const& pos, qreal r, qreal minMass, qreal maxMass, CellCluster* exclude) const = 0;
	using CellSelectFunction = bool(*)(Cell*);
	virtual QList< Cell* > getNearbySpecificCells(QVector2D const& pos, qreal r, CellSelectFunction selection) const = 0;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream, QMap< quint64, Cell* > const& oldIdCellMap) = 0;

protected:
	Cell*** _cellGrid = nullptr;
};

/****************** inline methods ******************/

Cell * CellMap::getCellFast(IntVector2D const& intPos) const
{
	return _cellGrid[intPos.x][intPos.y];
}

#endif //CELLMAP_H