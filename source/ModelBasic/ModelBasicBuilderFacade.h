#pragma once

#include "ChangeDescriptions.h"

#include "Definitions.h"

class ModelBasicBuilderFacade
{
public:
	virtual ~ModelBasicBuilderFacade() = default;
	
	virtual Serializer* buildSerializer() const = 0;
	virtual SymbolTable* buildDefaultSymbolTable() const = 0;
	virtual SimulationParameters* buildDefaultSimulationParameters() const = 0;
	virtual DescriptionHelper* buildDescriptionHelper() const = 0;
	virtual CellComputerCompiler* buildCellComputerCompiler(SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
};

