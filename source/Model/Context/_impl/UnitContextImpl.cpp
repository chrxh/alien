#include "Base/NumberGenerator.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/MapCompartment.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/Entities/CellCluster.h"
#include "Model/Entities/EnergyParticle.h"
#include "Model/Metadata/SymbolTable.h"

#include "UnitContextImpl.h"

UnitContextImpl::UnitContextImpl(QObject* parent)
	: UnitContext(parent)
{
}

UnitContextImpl::~UnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
}

void UnitContextImpl::init(NumberGenerator* numberGen, SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap
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

uint64_t UnitContextImpl::getTimestamp() const
{
	return _timestamp;
}

void UnitContextImpl::incTimestamp() 
{
	++_timestamp;
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

