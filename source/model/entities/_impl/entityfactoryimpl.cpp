#include "entityfactoryimpl.h"

#include "cellimpl.h"
#include "global/servicelocator.h"

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, Grid*& grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    return new CellImpl(energy, grid, maxConnections, tokenAccessNumber, relPos);
}

Cell* EntityFactoryImpl::buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , Grid*& grid)
{
    return new CellImpl(stream, connectingCells, grid);
}

Cell* EntityFactoryImpl::buildCellWithRandomData (qreal energy, Grid*& grid)
{
    return new CellImpl(energy, grid, true);
}

EntityFactoryImpl entityFactoryImpl;
