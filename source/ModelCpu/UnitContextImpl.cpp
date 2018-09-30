#include "Base/NumberGenerator.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/SymbolTable.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/ParticleMap.h"
#include "Model/Local/MapCompartment.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Particle.h"

#include "UnitContextImpl.h"

UnitContextImpl::UnitContextImpl(QObject* parent)
	: UnitContext(parent)
{
}

UnitContextImpl::~UnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
	delete _simulationParameters;
}

void UnitContextImpl::init(NumberGenerator* numberGen, SpacePropertiesLocal* metric, CellMap* cellMap, ParticleMap* energyMap
	, MapCompartment* mapCompartment, SimulationParameters* parameters)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_cellMap, cellMap);
	SET_CHILD(_energyParticleMap, energyMap);
	SET_CHILD(_mapCompartment, mapCompartment);

	delete _simulationParameters;	//no SET_CHILD here because thread of "_simulationParameters" may differs from that of "parameters"
	_simulationParameters = parameters;
	_simulationParameters->moveToThread(this->thread());

	deleteClustersAndEnergyParticles();
}

NumberGenerator * UnitContextImpl::getNumberGenerator() const
{
	return _numberGen;
}

SpacePropertiesLocal* UnitContextImpl::getSpaceProperties () const
{
    return _metric;
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

SimulationParameters* UnitContextImpl::getSimulationParameters() const
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

void UnitContextImpl::setSimulationParameters(SimulationParameters * parameters)
{
	delete _simulationParameters;	//no SET_CHILD here because thread of "_simulationParameters" may differs from that of "parameters"
	_simulationParameters = parameters;
	_simulationParameters->moveToThread(this->thread());
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

