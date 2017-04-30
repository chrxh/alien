#include "global/servicelocator.h"

#include "contextfactoryimpl.h"
#include "simulationunitcontextimpl.h"
#include "torustopologyimpl.h"
#include "mapcompartmentimpl.h"

namespace
{
	ContextFactoryImpl instance;
}

ContextFactoryImpl::ContextFactoryImpl()
{
	ServiceLocator::getInstance().registerService<ContextFactory>(this);
}

SimulationUnitContext * ContextFactoryImpl::buildSimulationUnitContext() const
{
	return new SimulationUnitContextImpl();
}

Topology * ContextFactoryImpl::buildTorusTopology() const
{
	return new TorusTopologyImpl();
}

MapCompartment * ContextFactoryImpl::buildMapCompartment() const
{
	return new MapCompartmentImpl();
}
