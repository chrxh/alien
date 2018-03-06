#pragma once

#include "Model/Api/SimulationContext.h"

class SimulationContextLocal
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextLocal(QObject* parent = nullptr) : SimulationContext(parent) {}
	virtual ~SimulationContextLocal() = default;

	virtual void init(NumberGenerator* numberGen, SpacePropertiesLocal* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters* parameters, CellComputerCompiler* compiler) = 0;

	virtual UnitGrid* getUnitGrid() const = 0;
	virtual UnitThreadController* getUnitThreadController() const = 0;
	virtual NumberGenerator* getNumberGenerator() const = 0;
};
