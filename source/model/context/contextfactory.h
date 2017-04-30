#ifndef CONTEXTFACTORY_H
#define CONTEXTFACTORY_H

#include "model/definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() {}

	virtual SimulationContext* buildSimulationContext(QObject* parent = nullptr) const = 0;
	virtual SimulationUnitContext* buildSimulationUnitContext(QObject* parent = nullptr) const = 0;
	virtual SimulationGrid* buildSimulationGrid(QObject* parent = nullptr) const = 0;
	virtual SimulationThreads* buildSimulationThreads(QObject* parent = nullptr) const = 0;
	virtual Topology* buildTorusTopology(QObject* parent = nullptr) const = 0;
	virtual MapCompartment* buildMapCompartment(QObject* parent = nullptr) const = 0;
};

#endif // CONTEXTFACTORY_H
