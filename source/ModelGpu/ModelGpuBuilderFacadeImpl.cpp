#include "Base/ServiceLocator.h"

#include "ModelBasic/SpaceProperties.h"

#include "SimulationControllerGpuImpl.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "SimulationMonitorGpuImpl.h"
#include "ModelGpuBuilderFacadeImpl.h"
#include "ModelGpuSettings.h"

SimulationControllerGpu * ModelGpuBuilderFacadeImpl::buildSimulationController(Config const & config, 
	ModelGpuData const & specificData, uint timestepAtBeginning) const
{
	auto context = new SimulationContextGpuImpl();

	SpaceProperties* spaceProp = new SpaceProperties();
	spaceProp->init(config.universeSize);
	context->init(spaceProp, timestepAtBeginning, config.symbolTable, config.parameters, specificData);

	auto controller = new SimulationControllerGpuImpl();
	controller->init(context);
	return controller;
}

SimulationAccessGpu * ModelGpuBuilderFacadeImpl::buildSimulationAccess() const
{
	return new SimulationAccessGpuImpl();
}

SimulationMonitorGpu * ModelGpuBuilderFacadeImpl::buildSimulationMonitor() const
{
	return new SimulationMonitorGpuImpl();
}

CudaConstants ModelGpuBuilderFacadeImpl::getDefaultCudaConstants() const
{
    return ModelGpuSettings::getDefaultCudaConstants();
}
