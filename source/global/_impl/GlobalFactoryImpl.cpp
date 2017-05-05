#include "global/ServiceLocator.h"

#include "GlobalFactoryImpl.h"
#include "RandomNumberGeneratorImpl.h"

namespace
{
	const int ARRAY_SIZE_FOR_RANDOM_NUMBERS = 234327;
}

namespace {
	GlobalFactoryImpl instance;
}

GlobalFactoryImpl::GlobalFactoryImpl()
{
	ServiceLocator::getInstance().registerService<GlobalFactory>(this);
}

RandomNumberGenerator * GlobalFactoryImpl::buildRandomNumberGenerator() const
{
	return new RandomNumberGeneratorImpl(ARRAY_SIZE_FOR_RANDOM_NUMBERS);
}
