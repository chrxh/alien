#include "global/RandomNumberGenerator.h"
#include "model/context/CellMap.h"
#include "model/context/EnergyParticleMap.h"
#include "model/context/MapCompartment.h"
#include "model/context/SimulationParameters.h"
#include "model/entities/CellCluster.h"
#include "model/entities/EnergyParticle.h"
#include "model/metadata/SymbolTable.h"

#include "UnitContextImpl.h"

UnitContextImpl::UnitContextImpl(QObject* parent)
	: UnitContext(parent)
{
}

UnitContextImpl::~UnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
}

void UnitContextImpl::init(RandomNumberGenerator* randomGen, SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap
	, MapCompartment* mapCompartment, SymbolTable* symbolTable, SimulationParameters* parameters)
{
	SET_CHILD(_randomGen, randomGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_cellMap, cellMap);
	SET_CHILD(_energyParticleMap, energyMap);
	SET_CHILD(_mapCompartment, mapCompartment);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);

	deleteClustersAndEnergyParticles();
}

RandomNumberGenerator * UnitContextImpl::getRandomNumberGenerator() const
{
	return _randomGen;
}

SpaceMetric* UnitContextImpl::getSpaceMetric () const
{
    return _metric;
}

EnergyParticleMap* UnitContextImpl::getEnergyParticleMap () const
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

QList<CellCluster*>& UnitContextImpl::getClustersRef ()
{
    return _clusters;
}

QList<EnergyParticle*>& UnitContextImpl::getEnergyParticlesRef ()
{
    return _energyParticles;
}

void UnitContextImpl::deleteClustersAndEnergyParticles()
{
	foreach(CellCluster* cluster, _clusters) {
		delete cluster;
	}
	_clusters.clear();
	foreach(EnergyParticle* particle, _energyParticles) {
		delete particle;
	}
	_energyParticles.clear();
}

