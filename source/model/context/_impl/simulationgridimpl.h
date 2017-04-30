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

	virtual void deleteUnits() override;
	virtual void registerUnit(IntVector2D gridPos, SimulationUnit* unit) override;

private:
	std::vector<std::vector<SimulationUnit*>> _units;
};

#endif // SIMULATIONGRIDIMPL_H
