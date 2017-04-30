#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include <QtGlobal>
#include <QVector3D>

#include "model/definitions.h"

class EntityFactory
{
public:
    virtual ~EntityFactory () {}

    virtual CellCluster* buildCellCluster (SimulationUnitContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, SimulationUnitContext* context) const = 0;
    virtual CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , SimulationUnitContext* context) const = 0;

    virtual Cell* buildCell (SimulationUnitContext* context) const = 0;
    virtual Cell* buildCell (qreal energy, SimulationUnitContext* context, int maxConnections
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const = 0;
    
    virtual Token* buildToken (SimulationUnitContext* context) const = 0;
	virtual Token* buildToken (SimulationUnitContext* context, qreal energy) const = 0;
	virtual Token* buildToken (SimulationUnitContext* context, qreal energy, QByteArray const& memory) const = 0;
	virtual Token* buildTokenWithRandomData(SimulationUnitContext* context, qreal energy) const = 0;

    virtual EnergyParticle* buildEnergyParticle(SimulationUnitContext* context) const = 0;
    virtual EnergyParticle* buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
        , SimulationUnitContext* context) const = 0;
};

#endif // ENTITYFACTORY_H
