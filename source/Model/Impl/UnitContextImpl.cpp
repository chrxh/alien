#include "Base/NumberGenerator.h"
#include "Model/Local/CellMap.h"
#include "Model/Local/ParticleMap.h"
#include "Model/Local/MapCompartment.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Particle.h"
#include "Model/Local/SymbolTable.h"

#include "UnitContextImpl.h"

UnitContextImpl::UnitContextImpl(QObject* parent)
	: UnitContext(parent)
{
}

UnitContextImpl::~UnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
}

void UnitContextImpl::init(NumberGenerator* numberGen, SpaceMetricLocal* metric, CellMap* cellMap, ParticleMap* energyMap
	, MapCompartment* mapCompartment, SymbolTable* symbolTable, SimulationParameters* parameters)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_cellMap, cellMap);
	SET_CHILD(_energyParticleMap, energyMap);
	SET_CHILD(_mapCompartment, mapCompartment);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);

	deleteClustersAndEnergyParticles();
}

NumberGenerator * UnitContextImpl::getNumberGenerator() const
{
	return _numberGen;
}

SpaceMetricLocal* UnitContextImpl::getSpaceMetric () const
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

SymbolTable* UnitContextImpl::getSymbolTable() const 
{
	return _symbolTable;
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

