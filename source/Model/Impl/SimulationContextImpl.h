#pragma once

#include "Model/Local/SimulationContextLocal.h"

class SimulationContextImpl
	: public SimulationContextLocal
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();

	virtual void init(NumberGenerator* numberGen, SpaceMetricLocal* metric, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters* parameters, CellComputerCompiler* compiler) override;

	virtual SpaceProperties* getSpaceMetric() const;
	virtual UnitGrid* getUnitGrid() const override;
	virtual UnitThreadController* getUnitThreadController() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;
	virtual NumberGenerator* getNumberGenerator() const override;
	virtual CellComputerCompiler* getCellComputerCompiler() const override;

private:
	NumberGenerator* _numberGen = nullptr;
	SpaceMetricLocal* _metric = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};

