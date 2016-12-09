#include "simulationcontextimpl.h"

#include "model/cellmap.h"

SimulationContextImpl::SimulationContextImpl()
{
}

SimulationContextImpl::~SimulationContextImpl ()
{

}

void SimulationContextImpl::lock ()
{
    _mutex.lock();
}

void SimulationContextImpl::unlock ()
{
    _mutex.unlock();
}

void SimulationContextImpl::init(IntVector2D size)
{
	foreach(CellCluster* cluster, _clusters) {
		delete cluster;
	}
	foreach(EnergyParticle* energy, _energyParticles) {
		delete energy;
	}
	_clusters.clear();
	_energyParticles.clear();

	_topology = Topology(size);
	_cellMap->init(&_topology);
}

Topology SimulationContextImpl::getTopology () const
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

void SimulationContextImpl::serialize(QDataStream & stream) const
{
}

void SimulationContextImpl::build(QDataStream & stream)
{
}

