#pragma once

#include "ModelBasic/SimulationContext.h"
#include "UnitObserver.h"

#include "Definitions.h"

class SimulationContextImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();

	virtual SpaceProperties* getSpaceProperties() const;
	virtual UnitGrid* getUnitGrid() const;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters * getSimulationParameters() const override;
	virtual CellComputerCompiler* getCellComputerCompiler() const override;

	virtual void setSimulationParameters(SimulationParameters* parameters) override;
	virtual NumberGenerator* getNumberGenerator() const override;

	virtual map<string, int> getSpecificData() const override;

	virtual void init(NumberGenerator* numberGen, SpacePropertiesImpl* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters * parameters, CellComputerCompiler* compiler);

	virtual UnitThreadController* getUnitThreadController() const;
	
private:
	NumberGenerator* _numberGen = nullptr;
	SpacePropertiesImpl* _metric = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SimulationAttributeSetter* _attributeSetter = nullptr;

	SymbolTable* _symbolTable = nullptr;
	SimulationParameters * _simulationParameters = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};

