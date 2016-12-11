#include "simulationcontextimpl.h"

#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"

SimulationContextImpl::SimulationContextImpl()
	: SimulationContextImpl({ 0, 0 })
{
}

SimulationContextImpl::SimulationContextImpl(IntVector2D size)
{
	_topology = new Topology(size);
	_energyParticleMap = new EnergyParticleMap(_topology);
	_cellMap = new CellMap(_topology);
}

SimulationContextImpl::~SimulationContextImpl ()
{
	deleteAttributes();
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

QList<CellCluster*>& SimulationContextImpl::getClustersRef ()
{
    return _clusters;
}

QList<EnergyParticle*>& SimulationContextImpl::getEnergyParticlesRef ()
{
    return _energyParticles;
}

void SimulationContextImpl::deleteAttributes()
{
	foreach(CellCluster* cluster, _clusters)
		delete cluster;
	foreach(EnergyParticle* energy, _energyParticles)
		delete energy;
	delete _cellMap;
	delete _energyParticleMap;
	delete _topology;
	_clusters.clear();
	_energyParticles.clear();
}

