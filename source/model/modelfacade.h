#ifndef MODELFACADE_H
#define MODELFACADE_H

#include "model/entities/cellto.h"
#include "model/features/constants.h"

#include <QtGlobal>
#include <QVector3D>

class Cell;
class Grid;

class ModelFacade
{
public:
    virtual ~ModelFacade () {}

    virtual Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, Grid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildFeaturedCell (qreal energy, CellFunctionType type, Grid*& grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildFeaturedCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, Grid*& grid) = 0;
    virtual Cell* buildFeaturedCell (QDataStream& stream, Grid*& grid) = 0;
    virtual Cell* buildFeaturedCellWithRandomData (qreal energy, Grid*& grid) = 0;
    virtual CellTO buildCellTO (Cell* cell) = 0;

    virtual void changeFeaturesOfCell (Cell* cell, CellFunctionType type, Grid*& grid) = 0;
};

#endif // MODELFACADE_H
