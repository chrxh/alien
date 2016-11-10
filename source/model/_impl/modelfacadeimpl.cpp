#include "modelfacadeimpl.h"

#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/entities/entityfactory.h"
#include "model/features/cellfunction.h"
#include "model/features/cellfunctioncomputer.h"
#include "model/features/energyguidance.h"
#include "model/features/cellfeaturefactory.h"
#include "global/servicelocator.h"
#include "model/simulationsettings.h"

ModelFacadeImpl::ModelFacadeImpl ()
{
    ServiceLocator::getInstance().registerService<ModelFacade>(this);
}

Cell* ModelFacadeImpl::buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, Grid*& grid
    , int maxConnections, int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellDecoratorFactory>();
    Cell* cell = entityFactory->buildCell(energy, grid, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, data, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

Cell* ModelFacadeImpl::buildDecoratedCell (qreal energy, CellFunctionType type, Grid*& grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellDecoratorFactory>();
    Cell* cell = entityFactory->buildCell(energy, grid, maxConnections, tokenAccessNumber, relPos);
    decoratorFactory->addCellFunction(cell, type, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

Cell* ModelFacadeImpl::buildDecoratedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , Grid*& grid)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    CellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<CellDecoratorFactory>();
    Cell* cell = entityFactory->buildCell(stream, connectingCells, grid);
    CellFunctionType type = CellFunction::getType(stream);
    decoratorFactory->addCellFunction(cell, type, stream, grid);
    decoratorFactory->addEnergyGuidance(cell, grid);
    return cell;
}

Cell* ModelFacadeImpl::buildDecoratedCell (QDataStream& stream, Grid*& grid)
{
    QMap< quint64, QList< quint64 > > temp;
    return buildDecoratedCell(stream, temp, grid);
}

Cell* ModelFacadeImpl::buildDecoratedCellWithRandomData (qreal energy, Grid*& grid)
{
    int randomMaxConnections = qrand() % (simulationParameters.MAX_CELL_CONNECTIONS+1);
    int randomTokenAccessNumber = qrand() % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
    quint8 randomData[256];
    for( int i = 0; i <256; ++i )
        randomData[i] = qrand()%256;
    CellFunctionType randomCellFunction = static_cast<CellFunctionType>(qrand() % static_cast<int>(CellFunctionType::_COUNTER));
    return buildDecoratedCell(energy, randomCellFunction, randomData, grid, randomMaxConnections, randomTokenAccessNumber, QVector3D());
}

CellTO ModelFacadeImpl::buildCellTO (Cell* cell)
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
    CellFunction* cellFunction = CellDecorator::findObject<CellFunction>(cell->getFeatureChain());
    to.cellFunctionType = cellFunction->getType();

    //copy computer data
    CellFunctionComputer* computer = CellDecorator::findObject<CellFunctionComputer>(cellFunction);
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

ModelFacadeImpl modelFacadeImpl;
