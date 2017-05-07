#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/EntityFactory.h"

class EntityFactoryImpl
	: public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

    CellCluster* buildCellCluster (UnitContext* context) const override;
    CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells, qreal angle, UnitContext* context) const override;
    Cell* buildCell (UnitContext* context) const override;
	Token* buildToken (UnitContext* context) const override;
	Token* buildTokenWithRandomData (UnitContext* context, qreal energy) const override;
    EnergyParticle* buildEnergyParticle(UnitContext* context) const override;

	virtual CellCluster* build(CellClusterDescription const& desc, UnitContext* context) const override;
	virtual Cell* build(CellDescription const& desc, UnitContext* context) const override;
	virtual Token* build(TokenDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* build(EnergyParticleDescription const& desc, UnitContext* context) const override;

	virtual CellCluster* build(CellClusterLightDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* build(EnergyParticleLightDescription const& desc, UnitContext* context) const override;
};

#endif // CELLFACTORYIMPL_H
