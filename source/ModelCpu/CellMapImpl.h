#pragma once

#include "CellMap.h"
#include "MapCompartment.h"
#include "UnitContext.h"

class CellMapImpl
	: public CellMap
{
	Q_OBJECT
public:
	CellMapImpl(QObject* parent = nullptr);
	virtual ~CellMapImpl();

	virtual void init(SpacePropertiesImpl* metric, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void setCell(QVector2D pos, Cell* cell) override;
	virtual void removeCellIfPresent(QVector2D pos, Cell* cellToRemove) override;
	virtual Cell* getCell(QVector2D pos) const override;

	//advanced functions
	virtual CellClusterSet getNearbyClusters(QVector2D const& pos, qreal r) const override;
	virtual Cluster* getNearbyClusterFast(QVector2D const& pos, qreal r, qreal minMass, qreal maxMass, Cluster* exclude) const override;
	using CellSelectFunction = bool(*)(Cell*);
	virtual QList< Cell* > getNearbySpecificCells(QVector2D const& pos, qreal r, CellSelectFunction selection) const override;

private:
	void deleteCellMap();
	inline Cell*& locateCell(IntVector2D & intPos) const;

	SpacePropertiesImpl* _metric = nullptr;
	MapCompartment* _compartment = nullptr;
	IntVector2D _size = { 0, 0 };
};

/****************** inline methods ******************/

Cell*& CellMapImpl::locateCell(IntVector2D & intPos) const
{
/*
	//TEMP
	return _cellGrid[intPos.x][intPos.y];
*/
	if (_compartment->isPointInCompartment(intPos)) {
		_compartment->convertAbsToRelPosition(intPos);
		return _cellGrid[intPos.x][intPos.y];
	}
	else {
		auto cellMap = static_cast<CellMapImpl*>(_compartment->getNeighborContext(intPos)->getCellMap());
		_compartment->convertAbsToRelPosition(intPos);
		return cellMap->_cellGrid[intPos.x][intPos.y];
	}
}
