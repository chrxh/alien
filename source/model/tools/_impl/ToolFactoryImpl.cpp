#include "global/ServiceLocator.h"

#include "MapManipulatorImpl.h"
#include "ToolFactoryImpl.h"

namespace
{
	ToolFactoryImpl instance;
}

ToolFactoryImpl::ToolFactoryImpl()
{
	ServiceLocator::getInstance().registerService<ToolFactory>(this);
}

MapManipulator * ToolFactoryImpl::buildMapManipulator() const
{
	return new MapManipulatorImpl();
}
