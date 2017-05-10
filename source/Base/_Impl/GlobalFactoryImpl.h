#ifndef GLOBALFACTORYIMPL_H
#define GLOBALFACTORYIMPL_H

#include "Base/GlobalFactory.h"

class GlobalFactoryImpl
	: public GlobalFactory
{
public:
	GlobalFactoryImpl();
	virtual ~GlobalFactoryImpl() = default;

	virtual NumberGenerator* buildRandomNumberGenerator() const override;
};

#endif // GLOBALFACTORYIMPL_H
