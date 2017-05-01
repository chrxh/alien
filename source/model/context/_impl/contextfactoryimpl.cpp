#include "global/servicelocator.h"

#include "contextfactoryimpl.h"
#include "simulationcontextimpl.h"
#include "simulationunitcontextimpl.h"
#include "simulationgridimpl.h"
#include "simulationthreadsimpl.h"
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

SimulationContext * ContextFactoryImpl::buildSimulationContext(QObject* parent) const
{
	return new SimulationContextImpl(parent);
}

SimulationUnitContext * ContextFactoryImpl::buildSimulationUnitContext(QObject* parent) const
{
	return new SimulationUnitContextImpl(parent);
}

SimulationUnit * ContextFactoryImpl::buildSimulationUnit(QObject * parent) const
{
	return new SimulationUnit(parent);
}

SimulationGrid * ContextFactoryImpl::buildSimulationGrid(QObject* parent) const
{
	return new SimulationGridImpl(parent);
}

SimulationThreads * ContextFactoryImpl::buildSimulationThreads(QObject* parent) const
{
	return new SimulationThreadsImpl(parent);
}

Topology * ContextFactoryImpl::buildTorusTopology(QObject* parent) const
{
	return new TorusTopologyImpl(parent);
}

MapCompartment * ContextFactoryImpl::buildMapCompartment(QObject* parent) const
{
	return new MapCompartmentImpl(parent);
}
