#pragma once

#include "ModelInterface/Definitions.h"
#include "ModelInterface/ChangeDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationAccess* buildSimulationAccess() const = 0;
};

