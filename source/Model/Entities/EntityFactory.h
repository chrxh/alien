#ifndef ENTITYFACTORY_H
#define ENTITYFACTORY_H

#include "Model/Definitions.h"
#include "ChangeDescriptions.h"

class EntityFactory
{
public:
	virtual ~EntityFactory() = default;

	virtual Cluster* build(ClusterChangeDescription const& desc, UnitContext* context) const = 0;
	virtual Cell* build(CellChangeDescription const& desc, UnitContext* context) const = 0;
	virtual Token* build(TokenDescription const& desc, UnitContext* context) const = 0;
	virtual Particle* build(ParticleChangeDescription const& desc, UnitContext* context) const = 0;
};

#endif // ENTITYFACTORY_H
