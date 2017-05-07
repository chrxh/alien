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
};

#endif // ACCESSPORTFACTORYIMPL_H
