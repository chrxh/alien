#pragma once

#include "ChangeDescriptions.h"

#include "Definitions.h"

class ModelBasicBuilderFacade
{
public:
	virtual ~ModelBasicBuilderFacade() = default;
	
	virtual Serializer* buildSerializer() const = 0;
	virtual DescriptionHelper* buildDescriptionHelper() const = 0;
	virtual CellComputerCompiler* buildCellComputerCompiler(SymbolTable* symbolTable, SimulationParameters const& parameters) const = 0;
    virtual SimulationChanger* buildSimulationChanger(SimulationMonitor* monitor, NumberGenerator* numberGenerator) const = 0;

	virtual SymbolTable* getDefaultSymbolTable() const = 0;
	virtual SimulationParameters getDefaultSimulationParameters() const = 0;
    virtual ExecutionParameters getDefaultExecutionParameters() const = 0;
};

