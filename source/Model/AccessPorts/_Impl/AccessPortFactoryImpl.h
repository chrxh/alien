#ifndef ACCESSPORTFACTORYIMPL_H
#define ACCESSPORTFACTORYIMPL_H

#include "Model/AccessPorts/AccessPortFactory.h"

class AccessPortFactoryImpl
	: public AccessPortFactory
{
public:
	virtual ~AccessPortFactoryImpl() = default;

	virtual SimulationAccess* buildSimulationAccess() const override;
};

#endif // ACCESSPORTFACTORYIMPL_H
