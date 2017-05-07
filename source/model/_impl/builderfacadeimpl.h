#ifndef BUILDERFACADEIMPL_H
#define BUILDERFACADEIMPL_H

#include "model/BuilderFacade.h"

class BuilderFacadeImpl
	: public BuilderFacade
{
public:
    BuilderFacadeImpl ();
	virtual ~BuilderFacadeImpl() = default;

	virtual SimulationContextApi* buildSimulationContext(int maxRunngingThreads, IntVector2D gridSize, SpaceMetric* metric
		, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SimulationFullAccess* buildSimulationFullAccess(SimulationContextApi* context) const override;
	virtual SimulationLightAccess* buildSimulationLightAccess(SimulationContextApi* context) const override;
	virtual SimulationController* buildSimulationController(SimulationContextApi* context) const override;
	virtual SpaceMetric* buildSpaceMetric(IntVector2D universeSize) const override;
	virtual SymbolTable* buildDefaultSymbolTable() const override;
	virtual SimulationParameters* buildDefaultSimulationParameters() const override;

private:
	Unit* buildSimulationUnit(IntVector2D gridPos, SimulationContext* context) const;
};

#endif // BUILDERFACADEIMPL_H
