#ifndef MODELBUILDERFACADE_H
#define MODELBUILDERFACADE_H

#include "Model/Entities/CellTO.h"
#include "Model/Features/CellFeatureEnums.h"
#include "Model/Entities/Descriptions.h"

#include "Definitions.h"

class ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacade() = default;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
		, SymbolTable* symbolTable, SimulationParameters* parameters) const = 0;
	virtual SimulationAccess* buildSimulationAccess(SimulationContextApi* context) const = 0;
	virtual SymbolTable* buildDefaultSymbolTable() const = 0;
	virtual SimulationParameters* buildDefaultSimulationParameters() const = 0;
};

#endif // MODELBUILDERFACADE_H
