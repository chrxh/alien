#pragma once

#include "Model/Local/AccessPortFactory.h"

class AccessPortFactoryImpl
	: public AccessPortFactory
{
public:
	virtual ~AccessPortFactoryImpl() = default;

	virtual SimulationAccess* buildSimulationAccess() const override;
};
