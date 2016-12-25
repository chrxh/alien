#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    CellCluster* buildEmptyCellCluster (SimulationContext* context) override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, SimulationContext* context) override;
    CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , SimulationContext* context) override;

    Cell* buildEmptyCell (SimulationContext* context) override;
    Cell* buildCell (qreal energy, SimulationContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) override;
    Cell* buildCellWithRandomData (qreal energy, SimulationContext* context) override;

    Token* buildEmptyToken () override;
};

#endif // CELLFACTORYIMPL_H
