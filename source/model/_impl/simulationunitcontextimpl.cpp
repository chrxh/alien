#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/simulationparameters.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/metadata/symboltable.h"

#include "simulationunitcontextimpl.h"

SimulationUnitContextImpl::SimulationUnitContextImpl()
{
	_energyParticleMap = new EnergyParticleMap();
	_cellMap = new CellMap();
	_symbolTable = new SymbolTable();
	_simulationParameters = new SimulationParameters();
}

SimulationUnitContextImpl::SimulationUnitContextImpl(SymbolTable * symbolTable)
	: _symbolTable(symbolTable)
{
	_energyParticleMap = new EnergyParticleMap();
	_cellMap = new CellMap();
	_simulationParameters = new SimulationParameters();
}

SimulationUnitContextImpl::~SimulationUnitContextImpl ()
{
	deleteAll();
}

void SimulationUnitContextImpl::init(Topology* topology)
{
	_topology = topology;
	_energyParticleMap->init(_topology);
	_cellMap->init(_topology);
	deleteClustersAndEnergyParticles();
}

void SimulationUnitContextImpl::lock ()
{
    _mutex.lock();
}

void SimulationUnitContextImpl::unlock ()
{
    _mutex.unlock();
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

std::set<quint64> SimulationUnitContextImpl::getAllCellIds() const
{
	QList< quint64 > cellIds;
	foreach(CellCluster* cluster, _clusters) {
		cellIds << cluster->getCellIds();
	}
	std::list<quint64> cellIdsStdList = cellIds.toStdList();
	std::set<quint64> cellIdsStdSet(cellIdsStdList.begin(), cellIdsStdList.end());
	return cellIdsStdSet;
}

void SimulationUnitContextImpl::deleteAll()
{
	deleteClustersAndEnergyParticles();
	delete _cellMap;
	delete _energyParticleMap;
	delete _topology;
	delete _simulationParameters;
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

