#ifndef ACCESSPORTFACTORY_H
#define ACCESSPORTFACTORY_H

#include "model/Definitions.h"
#include "Descriptions.h"
#include "LightDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationFullAccess* buildSimulationFullAccess() const = 0;
	virtual SimulationLightAccess* buildSimulationLightAccess() const = 0;

	virtual CellCluster* buildFromDescription(CellClusterDescription const& desc, UnitContext* context) const = 0;
	virtual CellCluster* buildFromDescription(CellClusterLightDescription const& desc, UnitContext* context) const = 0;
	virtual EnergyParticle* buildFromDescription(EnergyParticleDescription const& desc, UnitContext* context) const = 0;
	virtual EnergyParticle* buildFromDescription(EnergyParticleLightDescription const& desc, UnitContext* context) const = 0;
};

#endif // ACCESSPORTFACTORY_H
