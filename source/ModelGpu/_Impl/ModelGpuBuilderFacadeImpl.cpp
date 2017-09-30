#include "Base/ServiceLocator.h"

#include "Model/Context/SpaceMetricLocal.h"
#include "Model/Context/ContextFactory.h"
#include "Model/ModelBuilderFacade.h"

#include "SimulationControllerGpuImpl.h"
#include "SimulationContextGpuImpl.h"
#include "SimulationAccessGpuImpl.h"
#include "ModelGpuBuilderFacadeImpl.h"

SimulationController * ModelGpuBuilderFacadeImpl::buildSimulationController(IntVector2D universeSize, SymbolTable * symbolTable, SimulationParameters * parameters) const
{
	auto context = new SimulationContextGpuImpl();
	auto contextFactory = ServiceLocator::getInstance().getService<ContextFactory>();

	SpaceMetricLocal* metric = static_cast<SpaceMetricLocal*>(contextFactory->buildSpaceMetric());
	metric->init(universeSize);
	context->init(metric, symbolTable, parameters);

	auto controller = new SimulationControllerGpuImpl();
	controller->init(context);
	return controller;
}

SimulationAccess * ModelGpuBuilderFacadeImpl::buildSimulationAccess(SimulationContext * context) const
{
	auto access = new SimulationAccessGpuImpl();
	access->init(context);
	return access;
}
