#include "Base/ServiceLocator.h"

#include "ModelBasic/ModelBasicBuilderFacade.h"

#include "SimulationControllerGpuImpl.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "ModelGpuBuilderFacadeImpl.h"

SimulationControllerGpu * ModelGpuBuilderFacadeImpl::buildSimulationController(Config const & config, 
	ModelGpuData const & specificData, uint timestepAtBeginning) const
{
	auto context = new SimulationContextGpuImpl();
	auto contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();

	SpaceProperties* spaceProp = new SpaceProperties();
	spaceProp->init(universeSize);
	context->init(spaceProp, symbolTable, parameters);

	auto controller = new SimulationControllerGpuImpl();
	controller->init(context);
	return controller;
}

SimulationAccessGpu * ModelGpuBuilderFacadeImpl::buildSimulationAccess() const
{
	auto access = new SimulationAccessGpuImpl();
	access->init(context);
	return access;
}
