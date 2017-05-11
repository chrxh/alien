#ifndef ACCESSPORTFACTORY_H
#define ACCESSPORTFACTORY_H

#include "model/Definitions.h"
#include "model/Entities/Descriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationAccess* buildSimulationAccess() const = 0;
};

#endif // ACCESSPORTFACTORY_H
