#include "model/context/cellmap.h"
#include "model/context/energyparticlemap.h"
#include "model/context/simulationparameters.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/metadata/symboltable.h"

#include "simulationunitcontextimpl.h"

SimulationUnitContextImpl::SimulationUnitContextImpl(QObject* parent)
	: SimulationUnitContext(parent)
{
	_energyParticleMap = new EnergyParticleMap(this);
	_cellMap = new CellMap(this);
}

SimulationUnitContextImpl::~SimulationUnitContextImpl ()
{
	deleteClustersAndEnergyParticles();
}

void SimulationUnitContextImpl::init(Topology* topology, SymbolTable * symbolTable, SimulationParameters* parameters)
{
	if (_topology != topology) {
		delete _topology;
		_topology = topology;
	}
	if (_symbolTable != symbolTable) {
		delete _symbolTable;
		_symbolTable = symbolTable;
	}
	if (_simulationParameters != parameters) {
		delete _simulationParameters;
		_simulationParameters = parameters;
	}

	_energyParticleMap->init(_topology);
	_cellMap->init(_topology);
	deleteClustersAndEnergyParticles();
}

Topology* SimulationUnitContextImpl::getTopology () const
{
    return _topology;
}

EnergyParticleMap* SimulationUnitContextImpl::getEnergyParticleMap () const
{
    return _energyParticleMap;
}

CellMap* SimulationUnitContextImpl::getCellMap () const
{
    return _cellMap;
}

SymbolTable* SimulationUnitContextImpl::getSymbolTable() const 
{
	return _symbolTable;
}

SimulationParameters* SimulationUnitContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

QList<CellCluster*>& SimulationUnitContextImpl::getClustersRef ()
{
    return _clusters;
}

QList<EnergyParticle*>& SimulationUnitContextImpl::getEnergyParticlesRef ()
{
    return _energyParticles;
}

void SimulationUnitContextImpl::deleteClustersAndEnergyParticles()
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

