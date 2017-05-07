#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "model/entities/EntityFactory.h"

class EntityFactoryImpl
	: public EntityFactory
{
public:

    EntityFactoryImpl ();
    ~EntityFactoryImpl () {}

	virtual CellCluster* build(CellClusterDescription const& desc, UnitContext* context) const override;
	virtual Cell* build(CellDescription const& desc, UnitContext* context) const override;
	virtual Token* build(TokenDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* build(EnergyParticleDescription const& desc, UnitContext* context) const override;

	virtual CellCluster* build(CellClusterLightDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* build(EnergyParticleLightDescription const& desc, UnitContext* context) const override;
};

#endif // CELLFACTORYIMPL_H
