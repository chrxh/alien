#include "global/ServiceLocator.h"

#include "SimulationManipulatorImpl.h"
#include "ToolFactoryImpl.h"

namespace
{
	ToolFactoryImpl instance;
}

ToolFactoryImpl::ToolFactoryImpl()
{
	ServiceLocator::getInstance().registerService<ToolFactory>(this);
}

SimulationManipulator * ToolFactoryImpl::buildSimulationManipulator() const
{
	return new SimulationManipulatorImpl();
}
