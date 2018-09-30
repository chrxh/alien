#pragma once

#include "Model/Api/Definitions.h"
#include "Model/Api/ChangeDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationAccess* buildSimulationAccess() const = 0;
};

