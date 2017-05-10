#include "Base/ServiceLocator.h"

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

SimulationContext * ContextFactoryImpl::buildSimulationContext() const
{
	return new SimulationContextImpl();
}

UnitContext * ContextFactoryImpl::buildSimulationUnitContext() const
{
	return new UnitContextImpl();
}

Unit * ContextFactoryImpl::buildSimulationUnit() const
{
	return new UnitImpl();
}

UnitGrid * ContextFactoryImpl::buildSimulationGrid() const
{
	return new UnitGridImpl();
}

UnitThreadController * ContextFactoryImpl::buildSimulationThreads() const
{
	return new UnitThreadControllerImpl();
}

SpaceMetric * ContextFactoryImpl::buildSpaceMetric() const
{
	return new SpaceMetricImpl();
}

MapCompartment * ContextFactoryImpl::buildMapCompartment() const
{
	return new MapCompartmentImpl();
}

CellMap * ContextFactoryImpl::buildCellMap() const
{
	return new CellMapImpl();
}

EnergyParticleMap * ContextFactoryImpl::buildEnergyParticleMap() const
{
	return new EnergyParticleMapImpl();
}
