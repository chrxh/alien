#include "global/ServiceLocator.h"
#include "model/AccessPorts/Descriptions.h"
#include "model/AccessPorts/LightDescriptions.h"
#include "SimulationAccessImpl.h"
#include "AccessPortsFactoryImpl.h"

namespace
{
	AccessPortsFactoryImpl instance;
}

AccessPortsFactoryImpl::AccessPortsFactoryImpl()
{
	ServiceLocator::getInstance().registerService<AccessPortsFactory>(this);
}

SimulationFullAccess * AccessPortsFactoryImpl::buildSimulationFullAccess(QObject* parent /*= nullptr*/) const
{
	return new SimulationAccessImpl<DataDescription>(parent);
}

SimulationLightAccess * AccessPortsFactoryImpl::buildSimulationLightAccess(QObject* parent /*= nullptr*/) const
{
	return new SimulationAccessImpl<DataLightDescription>(parent);
}
