#include "SimulationControllerGpuImpl.h"

#include "ModelGpuBuilderFacadeImpl.h"

SimulationController * ModelGpuBuilderFacadeImpl::buildSimulationController(IntVector2D universeSize, SymbolTable * symbolTable, SimulationParameters * parameters) const
{
	auto controller = new SimulationControllerGpuImpl();
	return controller;

}

SimulationAccess * ModelGpuBuilderFacadeImpl::buildSimulationAccess(SimulationContextApi * context) const
{
	return nullptr;
}
