#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    CellCluster* buildEmptyCellCluster (Grid* grid) override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel
        , QVector3D vel, Grid* grid) override;
    CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle, Grid* grid) override;

    Cell* buildEmptyCell (Grid* grid) override;
    Cell* buildCell (qreal energy, Grid* grid, int maxConnections = 0, int tokenAccessNumber = 0
        , QVector3D relPos = QVector3D()) override;
    Cell* buildCellWithRandomData (qreal energy, Grid* grid) override;

    Token* buildEmptyToken () override;
};

#endif // CELLFACTORYIMPL_H
