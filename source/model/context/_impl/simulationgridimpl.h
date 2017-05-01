#ifndef SIMULATIONGRIDIMPL_H
#define SIMULATIONGRIDIMPL_H

#include "model/context/simulationgrid.h"

class SimulationGridImpl
	: public SimulationGrid
{
public:
	SimulationGridImpl(QObject* parent = nullptr);
	virtual ~SimulationGridImpl() {}

	virtual void init(IntVector2D gridSize) override;

	virtual void registerUnit(IntVector2D gridPos, SimulationUnit* unit) override;
	virtual IntVector2D getSize() const override;

private:
	IntVector2D _size = { 0, 0 };
	std::vector<std::vector<SimulationUnit*>> _units;
};

#endif // SIMULATIONGRIDIMPL_H
