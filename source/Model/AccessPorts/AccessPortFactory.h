#ifndef ACCESSPORTFACTORY_H
#define ACCESSPORTFACTORY_H

#include "Model/Definitions.h"
#include "Model/Entities/ChangeDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationAccess* buildSimulationAccess() const = 0;
};

#endif // ACCESSPORTFACTORY_H
