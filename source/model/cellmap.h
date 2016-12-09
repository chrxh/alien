#ifndef CELLMAP_H
#define CELLMAP_H

#include "definitions.h"

class CellMap
{
public:
	virtual ~CellMap();

	void init(Topology* topo);
	void clear();

	void setCell(QVector3D pos, Cell* cell);
	void removeCell(QVector3D pos);
	void removeCellIfPresent(QVector3D pos, Cell* cell);
	Cell* getCell(QVector3D pos) const;
	inline Cell* getCellFast(IntVector2D const& intPos) const;

	//advanced functions
	CellClusterSet getNearbyClusters(QVector3D const& pos, qreal r) const;
	CellCluster* getNearbyClusterFast(QVector3D const& pos, qreal r, qreal minMass, qreal maxMass, CellCluster* exclude) const;
	using CellSelectFunction = bool(*)(Cell*);
	QList< Cell* > getNearbySpecificCells(QVector3D const& pos, qreal r, CellSelectFunction selection) const;

	void serialize(QDataStream& stream) const;
	void build(QDataStream& stream, QMap< quint64, Cell* > const& oldIdCellMap);

private:
	void deleteCellMap();
	inline void removeCellIfPresent(int const &x, int const &y, Cell* cell);

	Topology* _topo = nullptr;
	Cell*** _cellGrid = nullptr;
};

Cell * CellMap::getCellFast(IntVector2D const& intPos) const
{
	return _cellGrid[intPos.x][intPos.y];
}

void CellMap::removeCellIfPresent(int const &x, int const &y, Cell* cell)
{
	if (_cellGrid[x][y] == cell)
		_cellGrid[x][y] = 0;
}

#endif //CELLMAP_H