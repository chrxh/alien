#pragma once

#include "Definitions.h"
#include "ModelInterface/ModelBuilderFacade.h"

class ModelBuilderFacadeImpl
	: public ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacadeImpl() = default;

	virtual Serializer* buildSerializer() const override;
	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize
		, IntVector2D universeSize, SymbolTable * symbolTable, SimulationParameters* parameters
		, uint timestep) const override;
	virtual SimulationAccess* buildSimulationAccess() const override;
	virtual SimulationMonitor* buildSimulationMonitor() const override;
	virtual DescriptionHelper* buildDescriptionHelper() const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContextImpl* context) const;
};
