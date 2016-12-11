#ifndef FACTORYFACADEIMPL_H
#define FACTORYFACADEIMPL_H

#include "model/factoryfacade.h"

class FactoryFacadeImpl : public FactoryFacade
{
public:
    FactoryFacadeImpl ();
	~FactoryFacadeImpl() = default;

    CellCluster* buildEmptyCellCluster (Grid* grid);
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel , QVector3D vel, Grid* grid);
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, quint8* data, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCell (qreal energy, CellFunctionType type, Grid* grid, int maxConnections = 0
        , int tokenAccessNumber = 0 , QVector3D relPos = QVector3D());
    Cell* buildFeaturedCellWithRandomData (qreal energy, Grid* grid);
    CellTO buildFeaturedCellTO (Cell* cell);
    void changeFeaturesOfCell (Cell* cell, CellFunctionType type, Grid* grid);
};

#endif // FACTORYFACADEIMPL_H
