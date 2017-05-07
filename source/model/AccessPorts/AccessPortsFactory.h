#ifndef TOOLFACTORY_H
#define TOOLFACTORY_H

#include "model/Definitions.h"

class AccessPortsFactory
{
public:
	virtual ~AccessPortsFactory() = default;

	virtual SimulationFullAccess* buildSimulationFullAccess(QObject* parent = nullptr) const = 0;
	virtual SimulationLightAccess* buildSimulationLightAccess(QObject* parent = nullptr) const = 0;
};

#endif // TOOLFACTORY_H
