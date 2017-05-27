#include "Base/ServiceLocator.h"

#include "ModelGpuBuilderFacadeImpl.h"

namespace {
	ModelGpuBuilderFacadeImpl instance;
}

ModelGpuBuilderFacadeImpl::ModelGpuBuilderFacadeImpl()
{
	ServiceLocator::getInstance().registerService<ModelGpuBuilderFacade>(this);
}

SimulationController * ModelGpuBuilderFacadeImpl::buildSimulationController(IntVector2D universeSize, SymbolTable * symbolTable, SimulationParameters * parameters) const
{
	return nullptr;
}

SimulationAccess * ModelGpuBuilderFacadeImpl::buildSimulationAccess(SimulationContextApi * context) const
{
	return nullptr;
}
