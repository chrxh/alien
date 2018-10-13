#include "Base/ServiceLocator.h"

#include "ModelBasic/SpaceProperties.h"

#include "SimulationControllerGpuImpl.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "ModelGpuBuilderFacadeImpl.h"

SimulationControllerGpu * ModelGpuBuilderFacadeImpl::buildSimulationController(Config const & config, 
	ModelGpuData const & specificData, uint timestepAtBeginning) const
{
	auto context = new SimulationContextGpuImpl();

	SpaceProperties* spaceProp = new SpaceProperties();
	spaceProp->init(config.universeSize);
	context->init(spaceProp, config.symbolTable, config.parameters);

	auto controller = new SimulationControllerGpuImpl();
	controller->init(context);
	return controller;
}

SimulationAccessGpu * ModelGpuBuilderFacadeImpl::buildSimulationAccess() const
{
	return new SimulationAccessGpuImpl();
}
