#ifndef MODELGPUBUILDERFACADEIMPL_H
#define MODELGPUBUILDERFACADEIMPL_H

#include "ModelGpu/ModelGpuBuilderFacade.h"

class ModelGpuBuilderFacadeImpl
	: public ModelGpuBuilderFacade
{
public:
	ModelGpuBuilderFacadeImpl();
	virtual ~ModelGpuBuilderFacadeImpl() = default;

	virtual SimulationController* buildSimulationController(IntVector2D universeSize, SymbolTable* symbolTable, SimulationParameters* parameters) const override;
	virtual SimulationAccess* buildSimulationAccess(SimulationContextApi* context) const override;
};

#endif // MODELGPUBUILDERFACADEIMPL_H
