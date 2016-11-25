#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/factoryfacade.h"

class FactoryFacadeImpl : public FactoryFacade
{
public:
    FactoryFacadeImpl ();
    ~FactoryFacadeImpl () {}

    CellCluster* buildEmptyCellCluster (Grid* grid);
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel , QVector3D vel, Grid* grid);
    CellCluster* buildCellCluster (QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
        ,  QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, Grid* grid);

    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid);
    Cell* buildFeaturedCell (QDataStream& stream, Grid* grid);
    Cell* buildFeaturedCellWithRandomData (qreal energy, Grid* grid);
    CellTO buildFeaturedCellTO (Cell* cell);
    void changeFeaturesOfCell (Cell* cell, CellFunctionType type, Grid* grid);
    void serializeFeaturedCell (Cell* cell, QDataStream& stream);

};

#endif // FACTORYFACADEIMPL_H
