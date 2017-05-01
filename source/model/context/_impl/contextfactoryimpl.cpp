#include "global/servicelocator.h"

#include "contextfactoryimpl.h"
#include "simulationcontextimpl.h"
#include "simulationunitimpl.h"
#include "simulationunitcontextimpl.h"
#include "simulationgridimpl.h"
#include "simulationthreadsimpl.h"
#include "spacemetricimpl.h"
#include "mapcompartmentimpl.h"
#include "cellmapimpl.h"
#include "energyparticlemapimpl.h"

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
	return new SimulationUnitImpl(parent);
}

SimulationGrid * ContextFactoryImpl::buildSimulationGrid(QObject* parent) const
{
	return new SimulationGridImpl(parent);
}

SimulationThreads * ContextFactoryImpl::buildSimulationThreads(QObject* parent) const
{
	return new SimulationThreadsImpl(parent);
}

SpaceMetric * ContextFactoryImpl::buildSpaceMetric(QObject* parent) const
{
	return new SpaceMetricImpl(parent);
}

MapCompartment * ContextFactoryImpl::buildMapCompartment(QObject* parent) const
{
	return new MapCompartmentImpl(parent);
}

CellMap * ContextFactoryImpl::buildCellMap(QObject * parent) const
{
	return new CellMapImpl(parent);
}

EnergyParticleMap * ContextFactoryImpl::buildEnergyParticleMap(QObject * parent) const
{
	return new EnergyParticleMapImpl(parent);
}
