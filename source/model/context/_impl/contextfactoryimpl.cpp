#include "global/ServiceLocator.h"

#include "ContextFactoryImpl.h"
#include "SimulationContextImpl.h"
#include "UnitImpl.h"
#include "UnitContextImpl.h"
#include "UnitGridImpl.h"
#include "UnitThreadControllerImpl.h"
#include "SpaceMetricImpl.h"
#include "MapCompartmentImpl.h"
#include "CellMapImpl.h"
#include "EnergyParticleMapImpl.h"

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

UnitGrid * ContextFactoryImpl::buildSimulationGrid(QObject* parent) const
{
	return new UnitGridImpl(parent);
}

UnitThreadController * ContextFactoryImpl::buildSimulationThreads(QObject* parent) const
{
	return new UnitThreadControllerImpl(parent);
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
