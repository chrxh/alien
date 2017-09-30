#pragma once

#include "Model/Api/ModelBuilderFacade.h"

class ModelBuilderFacadeImpl
	: public ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacadeImpl() = default;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
		, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SimulationAccess* buildSimulationAccess(SimulationContext* contextApi) const override;
	virtual CellConnector* buildCellConnector(SimulationContext* contextApi) const override;

	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContextLocal* context) const;
};
