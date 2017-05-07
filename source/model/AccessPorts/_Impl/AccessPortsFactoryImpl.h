#ifndef TOOLFACTORYIMPL_H
#define TOOLFACTORYIMPL_H

#include "model/AccessPorts/AccessPortsFactory.h"

class AccessPortsFactoryImpl
	: public AccessPortsFactory
{
public:
	AccessPortsFactoryImpl();
	virtual ~AccessPortsFactoryImpl() = default;

	virtual SimulationAccess* buildSimulationAccess() const override;
};

#endif // TOOLFACTORYIMPL_H
