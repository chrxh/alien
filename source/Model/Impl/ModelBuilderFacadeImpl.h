#pragma once

#include "Model/Api/ModelBuilderFacade.h"

class ModelBuilderFacadeImpl
	: public ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacadeImpl() = default;

	virtual Serializer* buildSerializer() const override;
	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
		, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SimulationAccess* buildSimulationAccess(SimulationContext* contextApi = nullptr) const override;
	virtual DescriptionHelper* buildDescriptionHelper(SimulationContext* contextApi) const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContextLocal* context) const;
};
