#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

#include "model/definitions.h"

class EntityFactory
{
public:
    virtual ~EntityFactory () {}

    virtual CellCluster* buildEmptyCellCluster (SimulationContext* context) = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, SimulationContext* context) = 0;
    virtual CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , SimulationContext* context) = 0;

    virtual Cell* buildEmptyCell (SimulationContext* context) = 0;
    virtual Cell* buildCell (qreal energy, SimulationContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) = 0;
    virtual Cell* buildCellWithRandomData (qreal energy, SimulationContext* context) = 0;

    virtual Token* buildEmptyToken () = 0;
};

#endif // ENTITYFACTORY_H
