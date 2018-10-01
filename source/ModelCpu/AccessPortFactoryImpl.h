#pragma once

#include "AccessPortFactory.h"

class AccessPortFactoryImpl
	: public AccessPortFactory
{
public:
	virtual ~AccessPortFactoryImpl() = default;

	virtual SimulationAccess* buildSimulationAccess() const override;
};
