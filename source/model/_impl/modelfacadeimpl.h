#ifndef MODELFACTORYIMPL_H
#define MODELFACTORYIMPL_H

#include "model/modelfacade.h"

class ModelFacadeImpl : public ModelFacade
{
public:
    ModelFacadeImpl ();
    ~ModelFacadeImpl () {}

    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid* grid);
    Cell* buildFeaturedCell (QDataStream& stream, Grid* grid);
    Cell* buildFeaturedCellWithRandomData (qreal energy, Grid* grid);
    CellTO buildCellTO (Cell* cell);

    void changeFeaturesOfCell (Cell* cell, CellFunctionType type, Grid* grid);

    void serializeFeaturedCell (Cell* cell, QDataStream& stream);

};

#endif // MODELFACTORYIMPL_H
