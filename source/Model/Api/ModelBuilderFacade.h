#pragma once

#include "Model/Api/CellFeatureEnums.h"
#include "Model/Api/ChangeDescriptions.h"

#include "Definitions.h"

class ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacade() = default;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
		, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationAccess* buildSimulationAccess(SimulationContext* context) const = 0;

	virtual CellConnector* buildCellConnector(SimulationContext* context) const = 0;

	virtual SymbolTable* buildDefaultSymbolTable() const = 0;
	virtual SimulationParameters* buildDefaultSimulationParameters() const = 0;
};

