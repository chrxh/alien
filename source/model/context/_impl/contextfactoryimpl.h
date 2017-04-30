#ifndef CONTEXTFACTORYIMPL_H
#define CONTEXTFACTORYIMPL_H

#include "model/context/contextfactory.h"

class ContextFactoryImpl
	: public ContextFactory
{
public:
	ContextFactoryImpl();
	virtual ~ContextFactoryImpl() {}

	virtual SimulationUnitContext* buildSimulationUnitContext() const;
	virtual Topology* buildTorusTopology() const;
	virtual MapCompartment* buildMapCompartment() const;
};

#endif // CONTEXTFACTORYIMPL_H
