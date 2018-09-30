#include "SimulationAccessImpl.h"
#include "AccessPortFactoryImpl.h"

SimulationAccess * AccessPortFactoryImpl::buildSimulationAccess() const
{
	return new SimulationAccessImpl();
}


