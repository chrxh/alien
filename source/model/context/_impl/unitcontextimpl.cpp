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

void UnitContextImpl::init(SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap, MapCompartment* mapCompartment, SymbolTable* symbolTable
	, SimulationParameters* parameters)
{
	if (_metric != metric) {
		delete _metric;
		_metric = metric;
	}
	if (_cellMap != cellMap) {
		delete _cellMap;
		_cellMap = cellMap;
	}
	if (_energyParticleMap != energyMap) {
		delete _energyParticleMap;
		_energyParticleMap = energyMap;
	}
	if (_mapCompartment != mapCompartment) {
		delete _mapCompartment;
		_mapCompartment = mapCompartment;
	}
	if (_symbolTable != symbolTable) {
		delete _symbolTable;
		_symbolTable = symbolTable;
	}
	if (_simulationParameters != parameters) {
		delete _simulationParameters;
		_simulationParameters = parameters;
	}

	deleteClustersAndEnergyParticles();
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

