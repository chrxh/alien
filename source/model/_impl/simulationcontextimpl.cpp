#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/simulationparameters.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/metadata/symboltable.h"

#include "simulationcontextimpl.h"

SimulationContextImpl::SimulationContextImpl()
{
	_topology = new Topology();
	_energyParticleMap = new EnergyParticleMap(_topology);
	_cellMap = new CellMap(_topology);
	_symbolTable = new SymbolTable();
	_simulationParameters = new SimulationParameters();
}

SimulationContextImpl::SimulationContextImpl(SymbolTable * symbolTable)
	: _symbolTable(symbolTable)
{
	_topology = new Topology();
	_energyParticleMap = new EnergyParticleMap(_topology);
	_cellMap = new CellMap(_topology);
	_simulationParameters = new SimulationParameters();
}

SimulationContextImpl::~SimulationContextImpl ()
{
	deleteAll();
}

void SimulationContextImpl::init(IntVector2D size)
{
	_topology->init(size);
	_energyParticleMap->init();
	_cellMap->init();
	deleteClustersAndEnergyParticles();
}

void SimulationContextImpl::initWithoutTopology()
{
	_energyParticleMap->init();
	_cellMap->init();
	deleteClustersAndEnergyParticles();
}

void SimulationContextImpl::lock ()
{
    _mutex.lock();
}

void SimulationContextImpl::unlock ()
{
    _mutex.unlock();
}


Topology* SimulationContextImpl::getTopology () const
{
    return _topology;
}

EnergyParticleMap* SimulationContextImpl::getEnergyParticleMap () const
{
    return _energyParticleMap;
}

CellMap* SimulationContextImpl::getCellMap () const
{
    return _cellMap;
}

SymbolTable* SimulationContextImpl::getSymbolTable() const 
{
	return _symbolTable;
}

SimulationParameters* SimulationContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

QList<CellCluster*>& SimulationContextImpl::getClustersRef ()
{
    return _clusters;
}

QList<EnergyParticle*>& SimulationContextImpl::getEnergyParticlesRef ()
{
    return _energyParticles;
}

std::set<quint64> SimulationContextImpl::getAllCellIds() const
{
	QList< quint64 > cellIds;
	foreach(CellCluster* cluster, _clusters) {
		cellIds << cluster->getCellIds();
	}
	std::list<quint64> cellIdsStdList = cellIds.toStdList();
	std::set<quint64> cellIdsStdSet(cellIdsStdList.begin(), cellIdsStdList.end());
	return cellIdsStdSet;
}

void SimulationContextImpl::deleteAll()
{
	deleteClustersAndEnergyParticles();
	delete _cellMap;
	delete _energyParticleMap;
	delete _topology;
	delete _simulationParameters;
}

void SimulationContextImpl::deleteClustersAndEnergyParticles()
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

