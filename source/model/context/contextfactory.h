#ifndef CONTEXTFACTORY_H
#define CONTEXTFACTORY_H

#include "model/definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() {}

	virtual SimulationUnitContext* buildSimulationUnitContext() const = 0;
	virtual Topology* buildTorusTopology() const = 0;
	virtual MapCompartment* buildMapCompartment() const = 0;
};

#endif // CONTEXTFACTORY_H
