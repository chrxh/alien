#include "GlobalFactoryImpl.h"
#include "NumberGeneratorImpl.h"

GlobalFactoryImpl::GlobalFactoryImpl() = default;

NumberGenerator * GlobalFactoryImpl::buildRandomNumberGenerator() const
{
	return new NumberGeneratorImpl();
}
