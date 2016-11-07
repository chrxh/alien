#include "entityfactoryimpl.h"

#include "aliencellimpl.h"
#include "global/servicelocator.h"

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

AlienCell* EntityFactoryImpl::buildCellWithRandomData (qreal energy, AlienGrid*& grid)
{
    return new AlienCellImpl(energy, grid, true);
}

AlienCell* EntityFactoryImpl::buildCell (qreal energy, AlienGrid*& grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    return new AlienCellImpl(energy, grid, false, maxConnections, tokenAccessNumber, relPos);
}

AlienCell* EntityFactoryImpl::buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , AlienGrid*& grid)
{
    return new AlienCellImpl(stream, connectingCells, grid);
}

AlienCell* EntityFactoryImpl::buildCellWithoutConnectingCells (QDataStream& stream
    , AlienGrid*& grid)
{
    return new AlienCellImpl(stream, grid);
}

EntityFactoryImpl entityFactoryImpl;
