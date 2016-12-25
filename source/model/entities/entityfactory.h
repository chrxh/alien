#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

#include "model/definitions.h"

class EntityFactory
{
public:
    virtual ~EntityFactory () {}

    virtual CellCluster* buildCellCluster (SimulationContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, SimulationContext* context) const = 0;
    virtual CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , SimulationContext* context) const = 0;

    virtual Cell* buildCell (SimulationContext* context) const = 0;
    virtual Cell* buildCell (qreal energy, SimulationContext* context, int maxConnections
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const = 0;
    virtual Cell* buildCellWithRandomData (qreal energy, SimulationContext* context) const = 0;

    virtual Token* buildToken () const = 0;

    virtual EnergyParticle* buildEnergyParticle(SimulationContext* context) const = 0;
    virtual EnergyParticle* buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
        , SimulationContext* context) const = 0;
};

#endif // ENTITYFACTORY_H
