#include "SimulationAccessCpuImpl.h"
#include "AccessPortFactoryImpl.h"

SimulationAccessCpu * AccessPortFactoryImpl::buildSimulationAccess() const
{
	return new SimulationAccessCpuImpl();
}


