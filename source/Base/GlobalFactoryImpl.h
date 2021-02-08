#pragma once

#include "GlobalFactory.h"

class GlobalFactoryImpl
	: public GlobalFactory
{
public:
	GlobalFactoryImpl();
	virtual ~GlobalFactoryImpl() = default;

	virtual NumberGenerator* buildRandomNumberGenerator() const override;
};
