#include "ContextFactoryImpl.h"
#include "SimulationContextImpl.h"
#include "UnitImpl.h"
#include "UnitContextImpl.h"
#include "UnitGridImpl.h"
#include "UnitThreadControllerImpl.h"
#include "SpacePropertiesImpl.h"
#include "CellMapImpl.h"
#include "ParticleMapImpl.h"
#include "CellComputerCompilerImpl.h"

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

SpacePropertiesLocal * ContextFactoryImpl::buildSpaceMetric() const
{
	return new SpacePropertiesImpl();
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

CellComputerCompilerLocal* ContextFactoryImpl::buildCellComputerCompiler() const
{
	return new CellComputerCompilerImpl();
}
