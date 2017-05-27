#ifndef MODELGPUBUILDERFACADE_H
#define MODELGPUBUILDERFACADE_H

#include "Model/Definitions.h"
#include "Definitions.h"

class ModelGpuBuilderFacade
{
public:
	virtual ~ModelGpuBuilderFacade() = default;

	virtual SimulationController* buildSimulationController(IntVector2D universeSize, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationAccess* buildSimulationAccess(SimulationContextApi* context) const = 0;
};

#endif // MODELGPUBUILDERFACADE_H
