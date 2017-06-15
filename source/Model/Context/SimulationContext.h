#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include "Model/Context/SimulationContextApi.h"

class SimulationContext
	: public SimulationContextApi
{
	Q_OBJECT
public:
	SimulationContext(QObject* parent = nullptr) : SimulationContextApi(parent) {}
	virtual ~SimulationContext() = default;

	virtual void init(NumberGenerator* numberGen, SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters* parameters) = 0;

	virtual UnitGrid* getUnitGrid() const = 0;
	virtual UnitThreadController* getUnitThreadController() const = 0;
};

#endif // SIMULATIONCONTEXT_H
