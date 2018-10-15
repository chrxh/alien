#include "ContextFactoryImpl.h"
#include "SimulationContextCpuImpl.h"
#include "UnitImpl.h"
#include "UnitContextImpl.h"
#include "UnitGridImpl.h"
#include "UnitThreadControllerImpl.h"
#include "CellMapImpl.h"
#include "ParticleMapImpl.h"

SimulationContextCpuImpl * ContextFactoryImpl::buildSimulationContext() const
{
	return new SimulationContextCpuImpl();
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

