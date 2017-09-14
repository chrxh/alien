#include "SerializationFacadeImpl.h"

#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Particle.h"
#include "Model/Entities/Token.h"
#include "Model/Entities/EntityFactory.h"
#include "Model/Features/CellFunction.h"
#include "Model/Features/CellComputer.h"
#include "Model/Features/EnergyGuidance.h"
#include "Model/Features/CellFeatureFactory.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/Settings.h"
#include "Model/Context/ContextFactory.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/EnergyParticleMap.h"
#include "Model/Context/SpaceMetric.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/Context/_Impl/UnitContextImpl.h"
#include "Model/ModelBuilderFacade.h"

void SerializationFacadeImpl::serializeSimulationContext(UnitContext * context, QDataStream & stream) const
{
	context->getSpaceMetric()->serializePrimitives(stream);

	auto const& clusters = context->getClustersRef();
	quint32 numCluster = clusters.size();
	stream << numCluster;
	foreach(Cluster* cluster, clusters) {
		serializeCellCluster(cluster, stream);
	}

	auto const& energyParticles = context->getParticlesRef();
	quint32 numEnergyParticles = energyParticles.size();
	stream << numEnergyParticles;
	foreach(Particle* e, energyParticles) {
		serializeEnergyParticle(e, stream);
	}

	context->getCellMap()->serializePrimitives(stream);
	context->getEnergyParticleMap()->serializePrimitives(stream);
	context->getSymbolTable()->serializePrimitives(stream);
	context->getSimulationParameters()->serializePrimitives(stream);
}

void SerializationFacadeImpl::deserializeSimulationContext(UnitContext* prevContext, QDataStream & stream) const
{
/*	//mapping old ids to new ids
	QMap< quint64, quint64 > oldNewCellIdMap;
	QMap< quint64, quint64 > oldNewClusterIdMap;

	//mapping old ids to new entities
	QMap< quint64, Cell* > oldIdCellMap;
	QMap< quint64, EnergyParticle* > oldIdEnergyMap;

	//deserialize map size
	auto metric = prevContext->getTopology();
	if (!metric) {
		ContextFactory* factory = ServiceLocator::getInstance().getService<ContextFactory>();
		metric = factory->buildSpaceMetric();
	}
	metric->deserializePrimitives(stream);
	prevContext->init(metric);

	//deserialize clusters
	quint32 numCluster;
	stream >> numCluster;
	for (quint32 i = 0; i < numCluster; ++i) {
		CellCluster* cluster = deserializeCellCluster(stream, oldNewClusterIdMap, oldNewCellIdMap, oldIdCellMap, prevContext);
		prevContext->getClustersRef() << cluster;
	}

	//deserialize energy particles
	quint32 numEnergyParticles;
	stream >> numEnergyParticles;
	for (quint32 i = 0; i < numEnergyParticles; ++i) {
        EnergyParticle* e = deserializeEnergyParticle(stream, oldIdEnergyMap, prevContext);
		prevContext->getEnergyParticlesRef() << e;
	}

	//deserialize maps
	prevContext->getCellMap()->deserializePrimitives(stream, oldIdCellMap);
	prevContext->getEnergyParticleMap()->deserializePrimitives(stream, oldIdEnergyMap);
	prevContext->getSymbolTable()->deserializePrimitives(stream);
	prevContext->getSimulationParameters()->deserializePrimitives(stream);*/
}

void SerializationFacadeImpl::serializeSimulationParameters(SimulationParameters* parameters, QDataStream& stream) const
{
	parameters->serializePrimitives(stream);
}

SimulationParameters* SerializationFacadeImpl::deserializeSimulationParameters(QDataStream& stream) const
{
	SimulationParameters* parameters = new SimulationParameters();
	parameters->deserializePrimitives(stream);
	return parameters;
}

void SerializationFacadeImpl::serializeSymbolTable(SymbolTable* symbolTable, QDataStream& stream) const
{
	symbolTable->serializePrimitives(stream);
}

SymbolTable* SerializationFacadeImpl::deserializeSymbolTable(QDataStream& stream) const
{
	SymbolTable* symbolTable = new SymbolTable();
	symbolTable->deserializePrimitives(stream);
	return symbolTable;
}

void SerializationFacadeImpl::serializeCellCluster(Cluster* cluster, QDataStream& stream) const
{
	cluster->serializePrimitives(stream);
	QList<Cell*>& cells = cluster->getCellsRef();
    stream << static_cast<quint32>(cells.size());
    foreach (Cell* cell, cells) {
		serializeFeaturedCell(cell, stream);
	}
	ClusterMetadata meta = cluster->getMetadata();
	stream << meta.name;
}

Cluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream
    , QMap< quint64, quint64 >& oldNewClusterIdMap, QMap< quint64, quint64 >& oldNewCellIdMap
    , QMap< quint64, Cell* >& oldIdCellMap, UnitContext* context) const
{
    auto entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	auto tagGen = ServiceLocator::getInstance().getService<TagGenerator>();
	Cluster* cluster = entityFactory->build(ClusterDescription(), context);
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

        quint64 newId = context->getNumberGenerator()->getTag();
        oldNewCellIdMap[cell->getId()] = newId;
        oldIdCellMap[cell->getId()] = cell;
        cell->setId(newId);
    }
    quint64 oldClusterId = cluster->getId();

    //assigning new cluster id
    quint64 id = context->getNumberGenerator()->getTag();
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

	ClusterMetadata meta;
	stream >> meta.name;
	cluster->setMetadata(meta);

    return cluster;
}

Cluster* SerializationFacadeImpl::deserializeCellCluster(QDataStream& stream
    , UnitContext* context) const
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
	for (int i = 0; i < numToken; ++i) {
		serializeToken(cell->getToken(i), stream);
	}

	int numConnections = cell->getNumConnections();
	for (int i = 0; i < numConnections; ++i) {
		stream << cell->getConnection(i)->getId();
	}

	CellMetadata meta = cell->getMetadata();
	stream << meta.color << meta.computerSourcecode << meta.description << meta.name;
}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream
	, QMap< quint64, QList< quint64 > >& connectingCells, UnitContext* context) const
{
	EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
	CellFeatureFactory* featureFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
	//TODO: new serialization
	Cell* cell;// = entityFactory->build(CellDescription(), context);
	featureFactory->addEnergyGuidance(cell, context);

	cell->deserializePrimitives(stream);
	quint8 rawType;
	stream >> rawType;
	Enums::CellFunction::Type type = static_cast<Enums::CellFunction::Type>(rawType);
	CellFeature* feature = featureFactory->addCellFunction(cell, type, context);
	feature->deserializePrimitives(stream);

	SimulationParameters* parameters = context->getSimulationParameters();
	for (int i = 0; i < cell->getNumToken(); ++i) {
		Token* token = deserializeToken(stream, context);
		if (i < parameters->cellMaxToken)
			cell->setToken(i, token);
		else
			delete token;
	}

	quint64 id;
	for (int i = 0; i < cell->getNumConnections(); ++i) {
		stream >> id;
		connectingCells[cell->getId()] << id;
	}

	CellMetadata meta;
	stream >> meta.color >> meta.computerSourcecode >> meta.description >> meta.name;
	cell->setMetadata(meta);

	return cell;
}

Cell* SerializationFacadeImpl::deserializeFeaturedCell(QDataStream& stream, UnitContext* context) const
{
	QMap< quint64, QList< quint64 > > temp;
	Cell* cell = deserializeFeaturedCell(stream, temp, context);
	cell->setId(context->getNumberGenerator()->getTag());
	return cell;
}

void SerializationFacadeImpl::serializeToken(Token* token, QDataStream& stream) const
{
    token->serializePrimitives(stream);
}

Token* SerializationFacadeImpl::deserializeToken(QDataStream& stream, UnitContext* context) const
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    Token* token = entityFactory->build(TokenDescription(), context);
    token->deserializePrimitives(stream);
    return token;
}

void SerializationFacadeImpl::serializeEnergyParticle(Particle* particle, QDataStream& stream) const
{
	particle->serializePrimitives(stream);
	auto metadata = particle->getMetadata();
	stream << metadata.color;
}

Particle* SerializationFacadeImpl::deserializeEnergyParticle(QDataStream& stream
    , QMap< quint64, Particle* >& oldIdEnergyMap, UnitContext* context) const
{
    auto factory = ServiceLocator::getInstance().getService<EntityFactory>();
	auto tagGen = ServiceLocator::getInstance().getService<TagGenerator>();
	Particle* particle = factory->build(ParticleDescription(), context);
    particle->deserializePrimitives(stream);
	ParticleMetadata metadata;
	stream >> metadata.color;
	particle->setMetadata(metadata);
	oldIdEnergyMap[particle->getId()] = particle;
	particle->setId(context->getNumberGenerator()->getTag());
	return particle;
}

Particle* SerializationFacadeImpl::deserializeEnergyParticle(QDataStream& stream
	, UnitContext* context) const
{
	auto tagGen = ServiceLocator::getInstance().getService<TagGenerator>();
	QMap< quint64, Particle* > temp;
	Particle* particle = deserializeEnergyParticle(stream, temp, context);
	ParticleMetadata metadata;
	stream >> metadata.color;
	particle->setMetadata(metadata);
	particle->setId(context->getNumberGenerator()->getTag());
	return particle;
}
