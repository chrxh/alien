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
	context->getTopology()->serializePrimitives(stream);

	auto const& clusters = context->getClustersRef();
	quint32 numCluster = clusters.size();
	stream << numCluster;
	foreach(CellCluster* cluster, clusters)
		serializeCellCluster(cluster, stream);

	auto const& energyParticles = context->getEnergyParticlesRef();
	quint32 numEnergyParticles = energyParticles.size();
	stream << numEnergyParticles;
	foreach(EnergyParticle* e, energyParticles)
		e->serialize(stream);

	context->getCellMap()->serializePrimitives(stream);
	context->getEnergyParticleMap()->serializePrimitives(stream);
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
	context->getTopology()->deserializePrimitives(stream);
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
	context->getCellMap()->deserializePrimitives(stream, oldIdCellMap);
	context->getEnergyParticleMap()->deserializePrimitives(stream, oldIdEnergyMap);
	return context;
}

void SerializationFacadeImpl::serializeCellCluster(CellCluster* cluster, QDataStream& stream)
{
	cluster->serializePrimitives(stream);
	QList<Cell*>& cells = cluster->getCellsRef();
    stream << static_cast<quint32>(cells.size());
	foreach( Cell* cell, cells) {
		serializeFeaturedCell(cell, stream);
	}
}

CellCluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
	, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, SimulationContext* context)
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellCluster* cluster = entityFactory->buildEmptyCellCluster(context);
	cluster->deserializePrimitives(stream);

	//read data and reconstructing structures
	QMap< quint64, QList< quint64 > > connectingCells;
	QMap< quint64, Cell* > idCellMap;
    quint32 numCells(0);
	stream >> numCells;

	QList< Cell* >& cells = cluster->getCellsRef();
    for (quint32 i = 0; i < numCells; ++i) {
		Cell* cell = deserializeFeaturedCell(stream, connectingCells, context);
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

void SerializationFacadeImpl::serializeFeaturedCell(Cell* cell, QDataStream& stream)
{
	cell->serializePrimitives(stream);
	CellFeature* features = cell->getFeatures();
	CellFunction* cellFunction = features->findObject<CellFunction>();
	if (cellFunction) {
		stream << static_cast<quint8>(cellFunction->getType());
        cellFunction->serializePrimitives(stream);
    }

    int numToken = cell->getNumToken();
    for( int i = 0; i < numToken; ++i) {
        serializeToken(cell->getToken(i), stream);
	}

	int numConnections = cell->getNumConnections;
	for (int i = 0; i < numConnections; ++i) {
		stream << cell->getConnection(i)->getId();
	}
	
}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , SimulationContext* context)
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildEmptyCell(context);
    cell->deserializePrimitives(stream);
    quint8 rawType;
	stream >> rawType;
    CellFunctionType type = static_cast<CellFunctionType>(rawType);
	decoratorFactory->addEnergyGuidance(cell, context);
    decoratorFactory->addCellFunction(cell, type, stream, context);

    for (int i = 0; i < cell->getNumToken(); ++i) {
        Token* token = deserializeToken(stream, context);
        if (i < simulationParameters.CELL_TOKENSTACKSIZE)
            cell->setToken(i, token);
        else
            delete token;
    }

    quint64 id;
    for (int i = 0; i < cell->getNumConnections(); ++i) {
        stream >> id;
        connectingCells[cell->getId()] << id;
    }

	return cell;
}


Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, SimulationContext* context)
{
    QMap< quint64, QList< quint64 > > temp;
    return deserializeFeaturedCell(stream, temp, context);
}

void SerializationFacadeImpl::serializeToken(Token* token, QDataStream& stream)
{
    token->serializePrimitives(stream);
}

Token* SerializationFacadeImpl::deserializeToken(QDataStream& stream, SimulationContext* context)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    Token* token = entityFactory->buildEmptyToken(context);
    token->deserializePrimitives(stream);
    return token;
}
