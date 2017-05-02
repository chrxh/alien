#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/entityfactory.h"

class EntityFactoryImpl : public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    CellCluster* buildCellCluster (UnitContext* context) const override;
    CellCluster* buildCellCluster (QList< Cell* > cells, qreal angle, QVector3D pos
        , qreal angularVel, QVector3D vel, UnitContext* context) const override;
    CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle
        , UnitContext* context) const override;

    Cell* buildCell (UnitContext* context) const override;
    Cell* buildCell (qreal energy, UnitContext* context, int maxConnections = 0
        , int tokenAccessNumber = 0, QVector3D relPos = QVector3D()) const override;

	Token* buildToken (UnitContext* context) const override;
	Token* buildToken(UnitContext* context, qreal energy) const override;
	Token* buildToken (UnitContext* context, qreal energy, QByteArray const& memory) const override;
	Token* buildTokenWithRandomData (UnitContext* context, qreal energy) const override;

    EnergyParticle* buildEnergyParticle(UnitContext* context) const override;
    EnergyParticle* buildEnergyParticle(qreal energy, QVector3D pos, QVector3D vel
        , UnitContext* context) const override;
};

#endif // CELLFACTORYIMPL_H
