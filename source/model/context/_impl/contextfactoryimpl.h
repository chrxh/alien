#ifndef CONTEXTFACTORYIMPL_H
#define CONTEXTFACTORYIMPL_H

#include "model/context/contextfactory.h"

class ContextFactoryImpl
	: public ContextFactory
{
public:
	ContextFactoryImpl();
	virtual ~ContextFactoryImpl() {}

	virtual SimulationContext* buildSimulationContext(QObject* parent = nullptr) const override;
	virtual SimulationUnitContext* buildSimulationUnitContext(QObject* parent = nullptr) const override;
	virtual SimulationUnit* buildSimulationUnit(QObject* parent = nullptr) const override;
	virtual SimulationGrid* buildSimulationGrid(QObject* parent = nullptr) const override;
	virtual SimulationThreads* buildSimulationThreads(QObject* parent = nullptr) const override;
	virtual Topology* buildTorusTopology(QObject* parent = nullptr) const override;
	virtual MapCompartment* buildMapCompartment(QObject* parent = nullptr) const override;
};

#endif // CONTEXTFACTORYIMPL_H
