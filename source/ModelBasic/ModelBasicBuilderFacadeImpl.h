#pragma once
#include "ModelBasicBuilderFacade.h"

class ModelBasicBuilderFacadeImpl
	: public ModelBasicBuilderFacade
{
public:
	~ModelBasicBuilderFacadeImpl() = default;

	Serializer* buildSerializer() const override;
	SymbolTable* buildDefaultSymbolTable() const override;
	SimulationParameters* buildDefaultSimulationParameters() const override;
	DescriptionHelper* buildDescriptionHelper() const override;
	CellComputerCompiler* buildCellComputerCompiler(SymbolTable* symbolTable, SimulationParameters* parameters) const override;
};
