#include "modelfactoryimpl.h"

#include "model/entities/entityfactory.h"
#include "model/decorators/aliencelldecoratorfactory.h"
#include "global/servicelocator.h"

ModelFactoryImpl::ModelFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<AlienCellDecoratorFactory>(this);
}

AlienCell* ModelFactoryImpl::buildDecoratedCellWithRandomData (qreal energy, AlienGrid*& grid)
{
    int randomMaxConnections = qrand() % (simulationParameters.MAX_CELL_CONNECTIONS+1);
    int randomTokenAccessNumber = qrand() % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
    quint8 randomData[256];
    CellFunctionType randomCellFunction = static_cast<CellFunctionType>(qrand() % CellFunctionType::_COUNTER);
    return buildDecoratedCell(energy, randomCellFunction, randomData, grid, randomMaxConnections, randomTokenAccessNumber, QVector3D());
}

AlienCell* ModelFactoryImpl::buildDecoratedCell (qreal energy, CellFunctionType type, quint8* data, AlienGrid*& grid
    , int maxConnections, int tokenAccessNumber, QVector3D relPos)
{
    EntityFactory* entityFactory = ServiceLocator::getInstance().getService<EntityFactory>();
    AlienCellDecoratorFactory* decoratorFactory = ServiceLocator::getInstance().getService<AlienCellDecoratorFactory>();
    AlienCell* cell = entityFactory->buildCell(energy, grid, maxConnections, tokenAccessNumber, relPos);
    cell = decoratorFactory->addCellFunction(c, type, data, grid);
    cell = decoratorFactory->addEnergyGuidance(c, grid);
    return cell;
}


ModelFactoryImpl modelFactoryImpl;
