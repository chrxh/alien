#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

#include "model/Definitions.h"

class AccessPortsFactory
{
public:
	virtual ~AccessPortsFactory() = default;

	virtual SimulationAccess* buildSimulationAccess() const = 0;
};

#endif // TOOLFACTORY_H
