#ifndef SIMULATIONGRID_H
#define SIMULATIONGRID_H

#include "model/definitions.h"

class SimulationGrid
	: public QObject
{
	Q_OBJECT
public:
	SimulationGrid(QObject* parent) : QObject(parent) {}
	virtual ~SimulationGrid() {}

	virtual void init(IntVector2D gridSize, Topology* topology) = 0;

	virtual void registerUnit(IntVector2D gridPos, SimulationUnit* unit) = 0;
	virtual IntVector2D getSize() const = 0;
	virtual IntRect calcMapRect(IntVector2D gridPos) const = 0;
};

#endif // SIMULATIONGRID_H
