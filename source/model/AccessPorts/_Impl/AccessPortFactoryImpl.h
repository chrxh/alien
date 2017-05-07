#ifndef ACCESSPORTFACTORYIMPL_H
#define ACCESSPORTFACTORYIMPL_H

#include "model/AccessPorts/AccessPortFactory.h"

class AccessPortFactoryImpl
	: public AccessPortFactory
{
public:
	AccessPortFactoryImpl();
	virtual ~AccessPortFactoryImpl() = default;

	virtual SimulationFullAccess* buildSimulationFullAccess() const override;
	virtual SimulationLightAccess* buildSimulationLightAccess() const override;

	virtual CellCluster* buildFromDescription(CellClusterDescription const& desc, UnitContext* context) const override;
	virtual CellCluster* buildFromDescription(CellClusterLightDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* buildFromDescription(EnergyParticleDescription const& desc, UnitContext* context) const override;
	virtual EnergyParticle* buildFromDescription(EnergyParticleLightDescription const& desc, UnitContext* context) const override;
};

#endif // ACCESSPORTFACTORYIMPL_H
