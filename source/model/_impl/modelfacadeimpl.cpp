#include "modelfacadeimpl.h"

#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"
#include "model/entities/entityfactory.h"
#include "model/decorators/aliencellfunction.h"
#include "model/decorators/aliencellfunctioncomputer.h"
#include "model/decorators/alienenergyguidance.h"
#include "model/decorators/aliencelldecoratorfactory.h"
#include "global/servicelocator.h"
#include "model/simulationsettings.h"

ModelFacadeImpl::ModelFacadeImpl ()
{
    ServiceLocator::getInstance().registerService<ModelFacade>(this);
}

AlienCell* ModelFacadeImpl::buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid
    , int maxConnections, int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    AlienCellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<AlienCellDecoratorFactory>();
    AlienCell* cell = entityFactory->buildCell(energy, grid, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, data, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

AlienCell* ModelFacadeImpl::buildDecoratedCell (qreal energy, CellFunctionType type, AlienGrid*& grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    AlienCellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<AlienCellDecoratorFactory>();
    AlienCell* cell = entityFactory->buildCell(energy, grid, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

AlienCell* ModelFacadeImpl::buildDecoratedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , AlienGrid*& grid)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    AlienCellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<AlienCellDecoratorFactory>();
    AlienCell* cell = entityFactory->buildCell(stream, connectingCells, grid);
    CellFunctionType type = AlienCellFunction::getType(stream);
    decoratorFactory->addCellFunction(cell, type, stream, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

AlienCell* ModelFacadeImpl::buildDecoratedCell (QDataStream& stream, AlienGrid*& grid)
{
    QMap< quint64, QList< quint64 > > temp;
    return buildDecoratedCell(stream, temp, grid);
}

AlienCell* ModelFacadeImpl::buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid)
{
    int randomMaxConnections = qrand() % (simulationParameters.MAX_CELL_CONNECTIONS+1);
    int randomTokenAccessNumber = qrand() % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
    quint8 randomData[256];
    for( int i = 0; i <256; ++i )
        randomData[i] = qrand()%256;
    CellFunctionType randomCellFunction = static_cast<CellFunctionType>(qrand() % static_cast<int>(CellFunctionType::_COUNTER));
    return buildDecoratedCell(energy, randomCellFunction, randomData, grid, randomMaxConnections, randomTokenAccessNumber, QVector3D());
}

AlienCellTO ModelFacadeImpl::buildCellTO (AlienCell* cell)
{
    AlienCellTO to;

    //copy cell properties
    AlienCellCluster* cluster = cell->getCluster();
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
    AlienCellFunction* cellFunction = AlienCellDecorator::findObject<AlienCellFunction>(cell->getFeatureChain());
    to.cellFunctionType = cellFunction->getType();

    //copy computer data
    AlienCellFunctionComputer* computer = AlienCellDecorator::findObject<AlienCellFunctionComputer>(cellFunction);
    if( computer ) {
        QVector< quint8 > d = computer->getMemoryReference();
        for(int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i)
            to.computerMemory[i] = d[i];
        to.computerCode = computer->decompileInstructionCode();
    }

    //copy token data
    for(int i = 0; i < cell->getNumToken(); ++i) {
        AlienToken* token = cell->getToken(i);
        to.tokenEnergies << token->energy;
        QVector< quint8 > d(simulationParameters.TOKEN_MEMSIZE);
        for(int j = 0; j < simulationParameters.TOKEN_MEMSIZE; ++j)
            d[j] = token->memory[j];
        to.tokenData << d;
    }
    return to;
}

ModelFacadeImpl modelFacadeImpl;
