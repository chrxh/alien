#ifndef CELLMAPIMPL_H
#define CELLMAPIMPL_H

#include "model/context/cellmap.h"

class CellMapImpl
	: public CellMap
{
	Q_OBJECT
public:
	CellMapImpl(QObject* parent = nullptr);
	virtual ~CellMapImpl();

	virtual void init(SpaceMetric* topo, MapCompartment* compartment) override;
	virtual void clear() override;

	virtual void setCell(QVector3D pos, Cell* cell) override;
	virtual void removeCell(QVector3D pos) override;
	virtual void removeCellIfPresent(QVector3D pos, Cell* cell) override;
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
	inline void removeCellIfPresent(int const &x, int const &y, Cell* cell);

	SpaceMetric* _topo = nullptr;
	int _gridSize = 0;
};


void CellMapImpl::removeCellIfPresent(int const &x, int const &y, Cell* cell)
{
	if (_cellGrid[x][y] == cell)
		_cellGrid[x][y] = nullptr;
}

#endif //CELLMAPIMPL_H