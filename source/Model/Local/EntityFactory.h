#pragma once

#include "Model/Api/ChangeDescriptions.h"
#include "Definitions.h"

class EntityFactory
{
public:
	virtual ~EntityFactory() = default;

	virtual Cluster* build(ClusterDescription const& desc, UnitContext* context) const = 0;
	virtual Cell* build(CellDescription const& cellDesc, Cluster* cluster, UnitContext* context) const = 0;
	virtual Token* build(TokenDescription const& desc, UnitContext* context) const = 0;
	virtual Particle* build(ParticleDescription const& desc, UnitContext* context) const = 0;
};

