#include "global/servicelocator.h"

#include "contextfactoryimpl.h"
#include "simulationcontextimpl.h"
#include "unitimpl.h"
#include "unitcontextimpl.h"
#include "gridimpl.h"
#include "threadcontrollerimpl.h"
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

UnitContext * ContextFactoryImpl::buildSimulationUnitContext(QObject* parent) const
{
	return new UnitContextImpl(parent);
}

Unit * ContextFactoryImpl::buildSimulationUnit(QObject * parent) const
{
	return new UnitImpl(parent);
}

Grid * ContextFactoryImpl::buildSimulationGrid(QObject* parent) const
{
	return new GridImpl(parent);
}

ThreadController * ContextFactoryImpl::buildSimulationThreads(QObject* parent) const
{
	return new ThreadControllerImpl(parent);
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
