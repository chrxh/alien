#include "Base/NumberGenerator.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SpaceProperties.h"

#include "CellMap.h"
#include "ParticleMap.h"
#include "MapCompartment.h"
#include "Cluster.h"
#include "Particle.h"

#include "UnitContextImpl.h"

UnitContextImpl::UnitContextImpl(QObject* parent)
	: UnitContext(parent)
{
}

UnitContextImpl::~UnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
}

void UnitContextImpl::init(NumberGenerator* numberGen, SpaceProperties* spaceProperties, CellMap* cellMap, ParticleMap* energyMap
	, MapCompartment* mapCompartment, SimulationParameters const& parameters)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_spaceProperties, spaceProperties);
	SET_CHILD(_cellMap, cellMap);
	SET_CHILD(_energyParticleMap, energyMap);
	SET_CHILD(_mapCompartment, mapCompartment);

	_simulationParameters = parameters;

	deleteClustersAndEnergyParticles();
}

NumberGenerator * UnitContextImpl::getNumberGenerator() const
{
	return _numberGen;
}

SpaceProperties* UnitContextImpl::getSpaceProperties () const
{
    return _spaceProperties;
}

ParticleMap* UnitContextImpl::getParticleMap () const
{
    return _energyParticleMap;
}

MapCompartment * UnitContextImpl::getMapCompartment() const
{
	return _mapCompartment;
}

CellMap* UnitContextImpl::getCellMap () const
{
    return _cellMap;
}

SimulationParameters const& UnitContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

uint64_t UnitContextImpl::getTimestamp() const
{
	return _timestamp;
}

void UnitContextImpl::incTimestamp() 
{
	++_timestamp;
}

void UnitContextImpl::setSimulationParameters(SimulationParameters const& parameters)
{
	_simulationParameters = parameters;
}

QList<Cluster*>& UnitContextImpl::getClustersRef ()
{
    return _clusters;
}

QList<Particle*>& UnitContextImpl::getParticlesRef ()
{
    return _energyParticles;
}

void UnitContextImpl::deleteClustersAndEnergyParticles()
{
	foreach(Cluster* cluster, _clusters) {
		delete cluster;
	}
	_clusters.clear();
	foreach(Particle* particle, _energyParticles) {
		delete particle;
	}
	_energyParticles.clear();
}

