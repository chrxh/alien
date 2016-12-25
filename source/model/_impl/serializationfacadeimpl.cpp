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

namespace
{
	SerializationFacadeImpl serializationFacadeImpl;
}


SerializationFacadeImpl::SerializationFacadeImpl()
{
    ServiceLocator::getInstance().registerService<SerializationFacade>(this);
}

void SerializationFacadeImpl::serializeSimulationContext(SimulationContext * context
    , QDataStream & stream) const
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
        e->serializePrimitives(stream);

	context->getCellMap()->serializePrimitives(stream);
	context->getEnergyParticleMap()->serializePrimitives(stream);
}

SimulationContext * SerializationFacadeImpl::deserializeSimulationContext(QDataStream & stream) const
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
        EnergyParticle* e = deserializeEnergyParticle(stream, oldIdEnergyMap, context);
		context->getEnergyParticlesRef() << e;
	}

	//deserialize maps
	context->getCellMap()->deserializePrimitives(stream, oldIdCellMap);
	context->getEnergyParticleMap()->deserializePrimitives(stream, oldIdEnergyMap);
	return context;
}

void SerializationFacadeImpl::serializeCellCluster(CellCluster* cluster, QDataStream& stream) const
{
	cluster->serializePrimitives(stream);
	QList<Cell*>& cells = cluster->getCellsRef();
    stream << static_cast<quint32>(cells.size());
    foreach (Cell* cell, cells) {
		serializeFeaturedCell(cell, stream);
	}
}

CellCluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream
    , SimulationContext* context) const
{
    QMap< quint64, quint64 > oldNewClusterIdMap;
    QMap< quint64, quint64 > oldNewCellIdMap;
    QMap< quint64, Cell* > oldIdCellMap;

    return deserializeCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, context);
}

void SerializationFacadeImpl::serializeFeaturedCell(Cell* cell, QDataStream& stream) const
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

    int numConnections = cell->getNumConnections();
	for (int i = 0; i < numConnections; ++i) {
		stream << cell->getConnection(i)->getId();
	}
	}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, SimulationContext* context) const
{
    QMap< quint64, QList< quint64 > > temp;
    Cell* cell = deserializeFeaturedCell(stream, temp, context);
    cell->setId(GlobalFunctions::createNewTag());
    return cell;
}

void SerializationFacadeImpl::serializeEnergyParticle(EnergyParticle* particle, QDataStream& stream) const
{
    particle->serializePrimitives(stream);
}

EnergyParticle* SerializationFacadeImpl::deserializeEnergyParticle(QDataStream& stream
    , SimulationContext* context) const
{
    QMap< quint64, EnergyParticle* > temp;
    EnergyParticle* particle = deserializeEnergyParticle(stream, temp, context);
    particle->id = GlobalFunctions::createNewTag();
    return particle;
}

void SerializationFacadeImpl::serializeToken(Token* token, QDataStream& stream) const
{
    token->serializePrimitives(stream);
}

Token* SerializationFacadeImpl::deserializeToken(QDataStream& stream) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    Token* token = entityFactory->buildToken();
    token->deserializePrimitives(stream);
    return token;
}

CellCluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream
    , QMap< quint64, quint64 >& oldNewClusterIdMap, QMap< quint64, quint64 >& oldNewCellIdMap
    , QMap< quint64, Cell* >& oldIdCellMap, SimulationContext* context) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellCluster* cluster = entityFactory->buildCellCluster(context);
    cluster->deserializePrimitives(stream);

    //read data and reconstructing structures
    QMap< quint64, QList< quint64 > > connectingCells;
    QMap< quint64, Cell* > idCellMap;
    quint32 numCells(0);
    stream >> numCells;

    //assigning new cell ids
    QList< Cell* >& cells = cluster->getCellsRef();
    for (quint32 i = 0; i < numCells; ++i) {
        Cell* cell = deserializeFeaturedCell(stream, connectingCells, context);
        cell->setCluster(cluster);
        cells << cell;
        idCellMap[cell->getId()] = cell;

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

    //set cell connections
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


Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream
    , QMap< quint64, QList< quint64 > >& connectingCells, SimulationContext* context) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildCell(context);
    featureFactory->addEnergyGuidance(cell, context);

    cell->deserializePrimitives(stream);
    quint8 rawType;
    stream >> rawType;
    CellFunctionType type = static_cast<CellFunctionType>(rawType);
    CellFeature* feature = featureFactory->addCellFunction(cell, type, context);
    feature->deserializePrimitives(stream);

    for (int i = 0; i < cell->getNumToken(); ++i) {
        Token* token = deserializeToken(stream);
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

EnergyParticle* SerializationFacadeImpl::deserializeEnergyParticle(QDataStream& stream
    , QMap< quint64, EnergyParticle* > oldIdEnergyMap, SimulationContext* context) const
{
    EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
    EnergyParticle* particle = factory->buildEnergyParticle(context);
    particle->deserializePrimitives(stream);
    oldIdEnergyMap[particle->id] = particle;
    return particle;
}
