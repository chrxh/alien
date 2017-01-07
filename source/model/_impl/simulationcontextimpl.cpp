#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/metadatamanager.h"

#include "simulationcontextimpl.h"

SimulationContextImpl::SimulationContextImpl()
{
	_topology = new Topology();
	_energyParticleMap = new EnergyParticleMap(_topology);
	_cellMap = new CellMap(_topology);
	_meta = new MetadataManager();
}

SimulationContextImpl::~SimulationContextImpl ()
{
	deleteAttributes();
}

void SimulationContextImpl::init(IntVector2D size)
{
	_topology->init(size);
	_energyParticleMap->init();
	_cellMap->init();
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

MetadataManager * SimulationContextImpl::getMetadataManager() const
{
	return _meta;
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

