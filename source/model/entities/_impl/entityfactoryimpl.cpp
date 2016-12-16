#include "entityfactoryimpl.h"

#include "cellimpl.h"
#include "cellclusterimpl.h"

#include "global/servicelocator.h"

EntityFactoryImpl::EntityFactoryImpl ()
{
    ServiceLocator::getInstance().registerService<EntityFactory>(this);
}

CellCluster* EntityFactoryImpl::buildEmptyCellCluster (Grid* grid)
{
    return new CellClusterImpl(grid);
}

CellCluster* EntityFactoryImpl::buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
    , QVector3D vel, Grid* grid)
{
    return new CellClusterImpl(cells, angle, pos, angularVel, vel, grid);
}

CellCluster* EntityFactoryImpl::buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle, Grid* grid)
{
    return new CellClusterImpl(cells, angle, grid);
}

Cell* EntityFactoryImpl::buildCell (qreal energy, Grid* grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    return new CellImpl(energy, grid, maxConnections, tokenAccessNumber, relPos);
}

Cell* EntityFactoryImpl::buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells
    , Grid* grid)
{
    return new CellImpl(stream, connectingCells, grid);
}

Cell* EntityFactoryImpl::buildCellWithRandomData (qreal energy, Grid* grid)
{
    return new CellImpl(energy, grid, true);
}

EntityFactoryImpl entityFactoryImpl;
