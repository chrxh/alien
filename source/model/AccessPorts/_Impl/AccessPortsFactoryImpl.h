#ifndef TOOLFACTORYIMPL_H
#define TOOLFACTORYIMPL_H

#include "model/AccessPorts/AccessPortsFactory.h"

class AccessPortsFactoryImpl
	: public AccessPortsFactory
{
public:
	AccessPortsFactoryImpl();
	virtual ~AccessPortsFactoryImpl() = default;

	virtual SimulationFullAccess* buildSimulationFullAccess(QObject* parent = nullptr) const override;
	virtual SimulationLightAccess* buildSimulationLightAccess(QObject* parent = nullptr) const override;
};

#endif // TOOLFACTORYIMPL_H
