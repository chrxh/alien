#pragma once

#include "Model/Local/SimulationContextLocal.h"
#include "Model/Local/UnitObserver.h"

#include "Definitions.h"

class SimulationContextImpl
	: public SimulationContextLocal
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();

	virtual void init(NumberGenerator* numberGen, SpacePropertiesLocal* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters * parameters, CellComputerCompiler* compiler) override;

	virtual IntVector2D getGridSize() const override;
	virtual uint getMaxThreads() const override;
	virtual SpaceProperties* getSpaceProperties() const;
	virtual UnitGrid* getUnitGrid() const override;
	virtual UnitThreadController* getUnitThreadController() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters const* getSimulationParameters() const override;
	virtual NumberGenerator* getNumberGenerator() const override;
	virtual CellComputerCompiler* getCellComputerCompiler() const override;

	virtual void setSimulationParameters(SimulationParameters const* parameters) override;
	
	NumberGenerator* _numberGen = nullptr;
	SpacePropertiesLocal* _metric = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SimulationAttributeSetter* _attributeSetter = nullptr;

	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};

