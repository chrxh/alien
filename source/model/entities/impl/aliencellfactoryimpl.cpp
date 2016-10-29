#include "aliencellfactoryimpl.h"

#include "aliencellimpl.h"

AlienCell* AlienCellFactoryImpl::buildCellWithRandomData (qreal energy, AlienGrid*& grid)
{
    return new AlienCellImpl(energy, grid, true);
}

AlienCell* AlienCellFactoryImpl::buildCell (qreal energy, AlienGrid*& grid, int maxConnections
    , int tokenAccessNumber, QVector3D relPos)
{
    return new AlienCellImpl(energy, grid, false, maxConnections, tokenAccessNumber, relPos);
}

AlienCell* AlienCellFactoryImpl::buildCell (QDataStream& stream
    , QMap< quint64, QList< quint64 > >& connectingCells, AlienGrid*& grid)
{
    return new AlienCellImpl(stream, connectingCells, grid);
}

AlienCell* AlienCellFactoryImpl::buildCellWithoutConnectingCells (QDataStream& stream
    , AlienGrid*& grid)
{
    return new AlienCellImpl(stream, grid);
}
