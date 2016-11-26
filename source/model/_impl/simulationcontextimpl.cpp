#include "simulationcontextimpl.h"

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

void SimulationContextImpl::reinit (QSize size)
{

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

