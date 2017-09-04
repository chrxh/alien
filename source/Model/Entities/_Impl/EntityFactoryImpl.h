#ifndef CELLFACTORYIMPL_H
#define CELLFACTORYIMPL_H

#include "Model/Entities/EntityFactory.h"

class EntityFactoryImpl
	: public EntityFactory
{
public:

	virtual ~EntityFactoryImpl() = default;

	virtual Cluster* build(ClusterDescription const& desc, UnitContext* context) const override;
	virtual Cell* build(CellDescription const& desc, UnitContext* context) const override;
	virtual Token* build(TokenDescription const& desc, UnitContext* context) const override;
	virtual Particle* build(ParticleDescription const& desc, UnitContext* context) const override;
};

#endif // CELLFACTORYIMPL_H
