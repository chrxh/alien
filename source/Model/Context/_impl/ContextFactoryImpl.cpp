#include "ContextFactoryImpl.h"
#include "SimulationContextImpl.h"
#include "UnitImpl.h"
#include "UnitContextImpl.h"
#include "UnitGridImpl.h"
#include "UnitThreadControllerImpl.h"
#include "SpaceMetricImpl.h"
#include "CellMapImpl.h"
#include "ParticleMapImpl.h"

SimulationContextLocal * ContextFactoryImpl::buildSimulationContext() const
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

SpaceMetricLocal * ContextFactoryImpl::buildSpaceMetric() const
{
	return new SpaceMetricImpl();
}

MapCompartment * ContextFactoryImpl::buildMapCompartment() const
{
	return new MapCompartment();
}

CellMap * ContextFactoryImpl::buildCellMap() const
{
	return new CellMapImpl();
}

ParticleMap * ContextFactoryImpl::buildEnergyParticleMap() const
{
	return new ParticleMapImpl();
}
