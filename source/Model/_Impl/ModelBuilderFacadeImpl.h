#ifndef BUILDERFACADEIMPL_H
#define BUILDERFACADEIMPL_H

#include "Model/ModelBuilderFacade.h"

class ModelBuilderFacadeImpl
	: public ModelBuilderFacade
{
public:
	virtual ~ModelBuilderFacadeImpl() = default;

	virtual SimulationController* buildSimulationController(int maxRunngingThreads, IntVector2D gridSize, IntVector2D universeSize
		, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SimulationAccess* buildSimulationAccess(SimulationContextApi* contextApi) const override;
	virtual CellConnector* buildCellConnector(SimulationContextApi* contextApi) const override;

	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const;
};

#endif // BUILDERFACADEIMPL_H
