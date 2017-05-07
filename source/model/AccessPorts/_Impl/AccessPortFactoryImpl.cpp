#include "global/ServiceLocator.h"
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

SimulationFullAccess * AccessPortFactoryImpl::buildSimulationFullAccess() const
{
	return new SimulationAccessImpl<DataDescription>();
}

SimulationLightAccess * AccessPortFactoryImpl::buildSimulationLightAccess() const
{
	return new SimulationAccessImpl<DataLightDescription>();
}

