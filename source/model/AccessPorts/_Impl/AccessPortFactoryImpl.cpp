#include "Base/ServiceLocator.h"
#include "SimulationAccessImpl.h"
#include "AccessPortFactoryImpl.h"

namespace
{
	AccessPortFactoryImpl instance;
}

AccessPortFactoryImpl::AccessPortFactoryImpl()
{
	ServiceLocator::getInstance().registerService<AccessPortFactory>(this);
}

SimulationAccess * AccessPortFactoryImpl::buildSimulationAccess() const
{
	return new SimulationAccessImpl();
}


