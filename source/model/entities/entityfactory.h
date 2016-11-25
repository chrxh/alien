#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

class Cell;
class CellCluster;
class Grid;

class EntityFactory
{
public:
    virtual ~EntityFactory () {}

    virtual CellCluster* buildEmptyCellCluster (Grid* grid);
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
        , QVector3D vel, Grid* grid) = 0;
    virtual CellCluster* buildCellCluster (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
        ,  QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid) = 0;
    virtual CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle, Grid* grid) = 0;

    virtual Cell* buildCell (qreal energy, Grid* grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid) = 0;
    virtual Cell* buildCellWithRandomData (qreal energy, Grid* grid) = 0;
};

#endif // ENTITYFACTORY_H
