#pragma once

#include "ModelBasic/SimulationContext.h"
#include "UnitObserver.h"

#include "Definitions.h"

class SimulationContextCpuImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextCpuImpl(QObject* parent = nullptr);
	virtual ~SimulationContextCpuImpl();

	virtual SpaceProperties* getSpaceProperties() const;
	virtual UnitGrid* getUnitGrid() const;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters const& getSimulationParameters() const override;
	virtual NumberGenerator* getNumberGenerator() const override;
    virtual int getTimestep() const override;
    virtual void setTimestep(int timestep) override;

	virtual void setSimulationParameters(SimulationParameters const& parameters) override;
    virtual void setExecutionParameters(ExecutionParameters const& parameters) override {}

	virtual map<string, int> getSpecificData() const override;

	virtual void init(SpaceProperties* spaceProp, UnitGrid* grid, UnitThreadController* threads
		, SymbolTable * symbolTable, SimulationParameters const& parameters, CellComputerCompiler* compiler);

	virtual UnitThreadController* getUnitThreadController() const;
	
private:
	SpaceProperties* _spaceProp = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SimulationAttributeSetter* _attributeSetter = nullptr;

	SymbolTable* _symbolTable = nullptr;
	SimulationParameters _simulationParameters;
	CellComputerCompiler* _compiler = nullptr;
	NumberGenerator* _numberGen = nullptr;
};

