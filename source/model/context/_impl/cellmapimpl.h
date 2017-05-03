#ifndef CELLMAPIMPL_H
#define CELLMAPIMPL_H

#include "model/context/CellMap.h"
#include "model/context/MapCompartment.h"
#include "model/context/UnitContext.h"

class CellMapImpl
	: public CellMap
{
	Q_OBJECT
public:
	CellMapImpl(QObject* parent = nullptr);
	virtual ~CellMapImpl();

	virtual void init(SpaceMetric* metric, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void setCell(QVector3D pos, Cell* cell) override;
	virtual void removeCellIfPresent(QVector3D pos, Cell* cellToRemove) override;
	virtual Cell* getCell(QVector3D pos) const override;

	//advanced functions
	virtual CellClusterSet getNearbyClusters(QVector3D const& pos, qreal r) const override;
	virtual CellCluster* getNearbyClusterFast(QVector3D const& pos, qreal r, qreal minMass, qreal maxMass, CellCluster* exclude) const override;
	using CellSelectFunction = bool(*)(Cell*);
	virtual QList< Cell* > getNearbySpecificCells(QVector3D const& pos, qreal r, CellSelectFunction selection) const override;

	virtual void serializePrimitives(QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream, QMap< quint64, Cell* > const& oldIdCellMap) override;

private:
	void deleteCellMap();
	inline Cell*& locateCell(IntVector2D const& intPos) const;

	SpaceMetric* _metric = nullptr;
	MapCompartment* _compartment = nullptr;
	IntVector2D _size = { 0, 0 };
};

/****************** inline methods ******************/

Cell*& CellMapImpl::locateCell(IntVector2D const& intPos) const
{
	if (_compartment->isPointInCompartment(intPos)) {
		auto relPos = _compartment->convertAbsToRelPosition(intPos);
		return _cellGrid[relPos.x][relPos.y];
	}
	else {
		auto cellMap = static_cast<CellMapImpl*>(_compartment->getNeighborContext(intPos)->getCellMap());
		auto relPos = _compartment->convertAbsToRelPosition(intPos);
		return cellMap->_cellGrid[relPos.x][relPos.y];
	}
}

#endif //CELLMAPIMPL_H