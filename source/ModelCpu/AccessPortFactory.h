#pragma once

#include "ModelBasic/Definitions.h"
#include "ModelBasic/ChangeDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationAccessCpu* buildSimulationAccess() const = 0;
};

