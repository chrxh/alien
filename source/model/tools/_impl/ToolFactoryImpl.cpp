#include "global/ServiceLocator.h"

#include "SimulationAccessImpl.h"
#include "ToolFactoryImpl.h"

namespace
{
	ToolFactoryImpl instance;
}

ToolFactoryImpl::ToolFactoryImpl()
{
	ServiceLocator::getInstance().registerService<ToolFactory>(this);
}

SimulationAccess * ToolFactoryImpl::buildSimulationManipulator() const
{
	return new SimulationAccessImpl();
}
