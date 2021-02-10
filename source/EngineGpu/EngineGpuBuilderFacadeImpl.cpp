#include "Base/ServiceLocator.h"

#include "EngineInterface/SpaceProperties.h"

#include "SimulationControllerGpuImpl.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "SimulationMonitorGpuImpl.h"
#include "EngineGpuBuilderFacadeImpl.h"
#include "EngineGpuSettings.h"

SimulationControllerGpu * EngineGpuBuilderFacadeImpl::buildSimulationController(Config const & config, 
	EngineGpuData const & specificData, uint timestepAtBeginning) const
{
	auto context = new SimulationContextGpuImpl();

	SpaceProperties* spaceProp = new SpaceProperties();
	spaceProp->init(config.universeSize);
	context->init(spaceProp, timestepAtBeginning, config.symbolTable, config.parameters, specificData);

	auto controller = new SimulationControllerGpuImpl();
	controller->init(context);
	return controller;
}

SimulationAccessGpu * EngineGpuBuilderFacadeImpl::buildSimulationAccess() const
{
	return new SimulationAccessGpuImpl();
}

SimulationMonitorGpu * EngineGpuBuilderFacadeImpl::buildSimulationMonitor() const
{
	return new SimulationMonitorGpuImpl();
}

CudaConstants EngineGpuBuilderFacadeImpl::getDefaultCudaConstants() const
{
    return EngineGpuSettings::getDefaultCudaConstants();
}
