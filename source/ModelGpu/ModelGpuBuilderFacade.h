#pragma once

#include "ModelInterface/Definitions.h"
#include "Definitions.h"

class ModelGpuBuilderFacade
{
public:
	virtual ~ModelGpuBuilderFacade() = default;

	virtual SimulationController* buildSimulationController(IntVector2D universeSize, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationAccess* buildSimulationAccess(SimulationContext* context) const = 0;

};

