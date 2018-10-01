#pragma once

#include "ModelInterface/SimulationContext.h"
#include "UnitObserver.h"

#include "Definitions.h"

class SimulationContextImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();


	virtual IntVector2D getGridSize() const override;
	virtual uint getMaxThreads() const override;
	virtual SpaceProperties* getSpaceProperties() const;
	virtual UnitGrid* getUnitGrid() const;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters * getSimulationParameters() const override;
	virtual CellComputerCompiler* getCellComputerCompiler() const override;

	virtual void setSimulationParameters(SimulationParameters* parameters) override;

	virtual void init(NumberGenerator* numberGen, SpacePropertiesImpl* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters * parameters, CellComputerCompiler* compiler);

	virtual UnitThreadController* getUnitThreadController() const;
	virtual NumberGenerator* getNumberGenerator() const;
	
	NumberGenerator* _numberGen = nullptr;
	SpacePropertiesImpl* _metric = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SimulationAttributeSetter* _attributeSetter = nullptr;

	SymbolTable* _symbolTable = nullptr;
	SimulationParameters * _simulationParameters = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};

