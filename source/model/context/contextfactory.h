#ifndef CONTEXTFACTORY_H
#define CONTEXTFACTORY_H

#include "model/definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() {}

	virtual SimulationContext* buildSimulationContext(QObject* parent = nullptr) const = 0;
	virtual SimulationUnitContext* buildSimulationUnitContext(QObject* parent = nullptr) const = 0;
	virtual SimulationUnit* buildSimulationUnit(QObject* parent = nullptr) const = 0;
	virtual SimulationGrid* buildSimulationGrid(QObject* parent = nullptr) const = 0;
	virtual ThreadController* buildSimulationThreads(QObject* parent = nullptr) const = 0;
	virtual SpaceMetric* buildSpaceMetric(QObject* parent = nullptr) const = 0;
	virtual MapCompartment* buildMapCompartment(QObject* parent = nullptr) const = 0;
	virtual CellMap* buildCellMap(QObject* parent = nullptr) const = 0;
	virtual EnergyParticleMap* buildEnergyParticleMap(QObject* parent = nullptr) const = 0;
};

#endif // CONTEXTFACTORY_H
