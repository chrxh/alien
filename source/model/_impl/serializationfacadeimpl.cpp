#include "serializationfacadeimpl.h"

#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/energyparticle.h"
#include "model/entities/token.h"
#include "model/entities/entityfactory.h"
#include "model/features/cellfunction.h"
#include "model/features/cellfunctioncomputer.h"
#include "model/features/energyguidance.h"
#include "model/features/cellfeaturefactory.h"
#include "model/simulationsettings.h"
#include "model/cellmap.h"
#include "model/energyparticlemap.h"
#include "model/topology.h"
#include "model/_impl/simulationcontextimpl.h"
#include "global/servicelocator.h"
#include "global/global.h"

namespace {
	SerializationFacadeImpl serializationFacadeImpl;
}

SerializationFacadeImpl::SerializationFacadeImpl()
{
	ServiceLocator::getInstance().registerService<SerializationFacade>(this);
}


void SerializationFacadeImpl::serializeSimulationContext(SimulationContext * context, QDataStream & stream)
{
	context->getTopology()->serialize(stream);

	auto const& clusters = context->getClustersRef();
	quint32 numCluster = clusters.size();
	stream << numCluster;
	foreach(CellCluster* cluster, clusters)
		cluster->serializePrimitives(stream);

	auto const& energyParticles = context->getEnergyParticlesRef();
	quint32 numEnergyParticles = energyParticles.size();
	stream << numEnergyParticles;
	foreach(EnergyParticle* e, energyParticles)
		e->serialize(stream);

	context->getCellMap()->serialize(stream);
	context->getEnergyParticleMap()->serialize(stream);
}

SimulationContext * SerializationFacadeImpl::deserializeSimulationContext(QDataStream & stream)
{
	//mapping old ids to new ids
	QMap< quint64, quint64 > oldNewCellIdMap;
	QMap< quint64, quint64 > oldNewClusterIdMap;

	//mapping old ids to new entities
	QMap< quint64, Cell* > oldIdCellMap;
	QMap< quint64, EnergyParticle* > oldIdEnergyMap;

	//deserialize map size
	SimulationContext* context = new SimulationContextImpl();
	context->getTopology()->deserialize(stream);
	context->getCellMap()->topologyUpdated();
	context->getEnergyParticleMap()->topologyUpdated();

	//deserialize clusters
	quint32 numCluster;
	stream >> numCluster;
	for (quint32 i = 0; i < numCluster; ++i) {
		CellCluster* cluster = deserializeCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, context);
		context->getClustersRef() << cluster;
	}

	//deserialize energy particles
	quint32 numEnergyParticles;
	stream >> numEnergyParticles;
	for (quint32 i = 0; i < numEnergyParticles; ++i) {
		EnergyParticle* e = new EnergyParticle(stream, oldIdEnergyMap, context);
		context->getEnergyParticlesRef() << e;
	}

	//deserialize maps
	context->getCellMap()->deserialize(stream, oldIdCellMap);
	context->getEnergyParticleMap()->deserialize(stream, oldIdEnergyMap);
	return context;
}

void SerializationFacadeImpl::serializeFeaturedCell(Cell* cell, QDataStream& stream)
{
	cell->serialize(stream);
	CellFeature* features = cell->getFeatures();
	CellFunction* cellFunction = features->findObject<CellFunction>();
	if (cellFunction) {
		stream << static_cast<quint8>(cellFunction->getType());
	}
	cellFunction->serialize(stream);
}

CellCluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
	, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid)
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellCluster* cluster = entityFactory->buildEmptyCellCluster(grid);

	//read data and reconstructing structures
	QMap< quint64, QList< quint64 > > connectingCells;
	QMap< quint64, Cell* > idCellMap;
	int numCells(0);
	stream >> numCells;

	QList< Cell* >& cells = cluster->getCellsRef();
	for (int i = 0; i < numCells; ++i) {
		Cell* cell = deserializeFeaturedCell(stream, connectingCells, grid);
		cell->setCluster(cluster);
		cells << cell;
		idCellMap[cell->getId()] = cell;

		//assigning new cell id
		quint64 newId = GlobalFunctions::createNewTag();
		oldNewCellIdMap[cell->getId()] = newId;
		oldIdCellMap[cell->getId()] = cell;
		cell->setId(newId);
	}
	quint64 oldClusterId = cluster->getId();

	//assigning new cluster id
	quint64 id = GlobalFunctions::createNewTag();
	cluster->setId(id);
	oldNewClusterIdMap[oldClusterId] = id;

	QMapIterator< quint64, QList< quint64 > > it(connectingCells);
	while (it.hasNext()) {
		it.next();
		Cell* cell(idCellMap[it.key()]);
		QList< quint64 > cellIdList(it.value());
		int i(0);
		foreach(quint64 cellId, cellIdList) {
			cell->setConnection(i, idCellMap[cellId]);
			++i;
		}
	}
	cluster->updateTransformationMatrix();
	cluster->updateRelCoordinates();
	cluster->updateAngularMass();

	return cluster;
}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
	, Grid* grid)
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	Cell* cell = entityFactory->buildCell(stream, connectingCells, grid);
	quint8 rawType;
	stream >> rawType;
	CellFunctionType type = static_cast<CellFunctionType>(rawType);
	decoratorFactory->addEnergyGuidance(cell, grid);
	decoratorFactory->addCellFunction(cell, type, stream, grid);
	return cell;
}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, Grid* grid)
{
	QMap< quint64, QList< quint64 > > temp;
	return deserializeFeaturedCell(stream, temp, grid);
}

