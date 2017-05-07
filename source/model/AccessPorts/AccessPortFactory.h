#ifndef ACCESSPORTFACTORY_H
#define ACCESSPORTFACTORY_H

#include "model/Definitions.h"
#include "model/entities/Descriptions.h"
#include "model/entities/LightDescriptions.h"

class AccessPortFactory
{
public:
	virtual ~AccessPortFactory() = default;

	virtual SimulationFullAccess* buildSimulationFullAccess() const = 0;
	virtual SimulationLightAccess* buildSimulationLightAccess() const = 0;
};

#endif // ACCESSPORTFACTORY_H
