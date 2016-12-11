#ifndef FACTORYFACADE_H
#define FACTORYFACADE_H

#include "definitions.h"
#include "model/entities/cellto.h"
#include "model/features/cellfeatureconstants.h"

#include <QtGlobal>
#include <QVector3D>

class FactoryFacade
{
public:
    virtual ~FactoryFacade () {}

    virtual CellCluster* buildEmptyCellCluster (Grid* grid) = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel, Grid* grid) = 0;
    virtual Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildFeaturedCell (qreal energy, CellFunctionType type, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildFeaturedCellWithRandomData (qreal energy, Grid* grid) = 0;
    virtual CellTO buildFeaturedCellTO (Cell* cell) = 0;
    virtual void changeFeaturesOfCell (Cell* cell, CellFunctionType type, Grid* grid) = 0;
};

#endif // FACTORYFACADE_H
