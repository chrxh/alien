#include "global/ServiceLocator.h"

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

SimulationAccess * AccessPortsFactoryImpl::buildSimulationAccess() const
{
	return new SimulationAccessImpl();
}
