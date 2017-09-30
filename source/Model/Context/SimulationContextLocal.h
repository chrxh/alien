#pragma once

#include "Model/SimulationContext.h"

class SimulationContextLocal
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextLocal(QObject* parent = nullptr) : SimulationContext(parent) {}
	virtual ~SimulationContextLocal() = default;

	virtual void init(NumberGenerator* numberGen, SpaceMetricLocal* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters* parameters) = 0;

	virtual UnitGrid* getUnitGrid() const = 0;
	virtual UnitThreadController* getUnitThreadController() const = 0;
	virtual NumberGenerator* getNumberGenerator() const = 0;
};
