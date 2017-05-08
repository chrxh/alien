#ifndef UNITGRID_H
#define UNITGRID_H

#include "model/Definitions.h"

class UnitGrid
	: public QObject
{
	Q_OBJECT
public:
	UnitGrid(QObject* parent) : QObject(parent) {}
	virtual ~UnitGrid() = default;

	virtual void init(IntVector2D gridSize, SpaceMetric* metric) = 0;

	virtual void registerUnit(IntVector2D gridPos, Unit* unit) = 0;
	virtual IntVector2D getSize() const = 0;
	virtual Unit* getUnitOfGridPos(IntVector2D gridPos) const = 0;
	virtual Unit* getUnitOfMapPos(QVector2D pos) const = 0;
	virtual IntVector2D getGridPosOfMapPos(QVector2D pos) const = 0;
	virtual IntRect calcCompartmentRect(IntVector2D gridPos) const = 0;
};

#endif // UNITGRID_H
