#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    CellCluster* buildEmptyCellCluster (Grid* grid);
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
        , QVector3D vel, Grid* grid);
    CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle, Grid* grid);
    CellCluster* buildCellCluster (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
        ,  QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid);

    Cell* buildCell (qreal energy, Grid* grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D());
    Cell* buildCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid);
    Cell* buildCellWithRandomData (qreal energy, Grid* grid);
};

#endif // CELLFACTORYIMPL_H
