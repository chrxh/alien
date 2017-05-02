#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "model/definitions.h"

class EntityFactory
{
public:
    virtual ~EntityFactory () {}

    virtual CellCluster* buildCellCluster (UnitContext* context) const = 0;
    virtual CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, UnitContext* context) const = 0;
    virtual CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , UnitContext* context) const = 0;

    virtual Cell* buildCell (UnitContext* context) const = 0;
    virtual Cell* buildCell (qreal energy, UnitContext* context, int maxConnections
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const = 0;
    
    virtual Token* buildToken (UnitContext* context) const = 0;
	virtual Token* buildToken (UnitContext* context, qreal energy) const = 0;
	virtual Token* buildToken (UnitContext* context, qreal energy, QByteArray const& memory) const = 0;
	virtual Token* buildTokenWithRandomData(UnitContext* context, qreal energy) const = 0;

    virtual EnergyParticle* buildEnergyParticle(UnitContext* context) const = 0;
    virtual EnergyParticle* buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
        , UnitContext* context) const = 0;
};

#endif // ENTITYFACTORY_H
