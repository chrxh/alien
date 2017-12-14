#pragma once

#include "Model/Api/CellFeatureEnums.h"
#include "Model/Api/ChangeDescriptions.h"

#include "Definitions.h"

class ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacade() = default;
	
	virtual Serializer* buildSerializer() const = 0;
	virtual SymbolTable* buildDefaultSymbolTable() const = 0;
	virtual SimulationParameters* buildDefaultSimulationParameters() const = 0;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize
		, IntVector2D universeSize, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationAccess* buildSimulationAccess() const = 0;
	virtual DescriptionHelper* buildDescriptionHelper() const = 0;
};

