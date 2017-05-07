#include "global/ServiceLocator.h"

#include "GlobalFactoryImpl.h"
#include "NumberGeneratorImpl.h"

namespace {
	GlobalFactoryImpl instance;
}

GlobalFactoryImpl::GlobalFactoryImpl()
{
	ServiceLocator::getInstance().registerService<GlobalFactory>(this);
}

NumberGenerator * GlobalFactoryImpl::buildRandomNumberGenerator() const
{
	return new NumberGeneratorImpl();
}
