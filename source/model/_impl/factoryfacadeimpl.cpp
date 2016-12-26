#include "factoryfacadeimpl.h"

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

namespace {
	FactoryFacadeImpl factoryFacadeImpl;
}

FactoryFacadeImpl::FactoryFacadeImpl ()
{
    ServiceLocator::getInstance().registerService<FactoryFacade>(this);
}

CellCluster* FactoryFacadeImpl::buildCellCluster (SimulationContext* context)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    return entityFactory->buildCellCluster(context);
}

CellCluster* FactoryFacadeImpl::buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
    , QVector3D vel, SimulationContext* context)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    return entityFactory->buildCellCluster(cells, angle, pos, angularVel, vel, context);
}


Cell* FactoryFacadeImpl::buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data
    , SimulationContext* context, int maxConnections, int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildCell(energy, context, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, data, context);
    decoratorFactory->addEnergyGuidance(cell, context);
    return cell;
}

Cell* FactoryFacadeImpl::buildFeaturedCell (qreal energy, CellFunctionType type, SimulationContext* context
    , int maxConnections, int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    Cell* cell = entityFactory->buildCell(energy, context, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, context);
    decoratorFactory->addEnergyGuidance(cell, context);
    return cell;
}

Cell* FactoryFacadeImpl::buildFeaturedCellWithRandomData (qreal energy, SimulationContext* context)
{
    int randomMaxConnections = qrand() % (simulationParameters.MAX_CELL_CONNECTIONS+1);
    int randomTokenAccessNumber = qrand() % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
    quint8 randomData[256];
    for( int i = 0; i <256; ++i )
        randomData[i] = qrand()%256;
    CellFunctionType randomCellFunction = static_cast<CellFunctionType>(qrand() % static_cast<int>(CellFunctionType::_COUNTER));
    return buildFeaturedCell(energy, randomCellFunction, randomData, context, randomMaxConnections, randomTokenAccessNumber, QVector3D());
}

CellTO FactoryFacadeImpl::buildFeaturedCellTO (Cell* cell)
{
    CellTO to;

    //copy cell properties
    CellCluster* cluster = cell->getCluster();
    to.numCells = cluster->getMass();
    to.clusterPos = cluster->getPosition();
    to.clusterVel = cluster->getVel();
    to.clusterAngle = cluster->getAngle();
    to.clusterAngVel = cluster->getAngularVel();
    to.cellPos = cell->calcPosition();
    to.cellEnergy = cell->getEnergy();
    to.cellNumCon = cell->getNumConnections();
    to.cellMaxCon = cell->getMaxConnections();
    to.cellAllowToken = !cell->isTokenBlocked();
    to.cellTokenAccessNum = cell->getTokenAccessNumber();
    CellFunction* cellFunction = cell->getFeatures()->findObject<CellFunction>();
    to.cellFunctionType = cellFunction->getType();

    //copy computer data
    CellFunctionComputer* computer = cellFunction->findObject<CellFunctionComputer>();
    if( computer ) {
        QVector< quint8 > d = computer->getMemoryReference();
        for(int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i)
            to.computerMemory[i] = d[i];
        to.computerCode = computer->decompileInstructionCode();
    }

    //copy token data
    for(int i = 0; i < cell->getNumToken(); ++i) {
        Token* token = cell->getToken(i);
        to.tokenEnergies << token->energy;
        QVector< quint8 > d(simulationParameters.TOKEN_MEMSIZE);
        for(int j = 0; j < simulationParameters.TOKEN_MEMSIZE; ++j)
            d[j] = token->memory[j];
        to.tokenData << d;
    }
    return to;
}

void FactoryFacadeImpl::changeFeaturesOfCell (Cell* cell, CellFunctionType type, SimulationContext* context)
{
    cell->removeFeatures();
    CellFeatureFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
    decoratorFactory->addCellFunction(cell, type, context);
    decoratorFactory->addEnergyGuidance(cell, context);
}

