#ifndef CONTEXTFACTORY_H
#define CONTEXTFACTORY_H

#include "model/Definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() {}

	virtual SimulationContext* buildSimulationContext(QObject* parent = nullptr) const = 0;
	virtual UnitContext* buildSimulationUnitContext(QObject* parent = nullptr) const = 0;
	virtual Unit* buildSimulationUnit(QObject* parent = nullptr) const = 0;
	virtual UnitGrid* buildSimulationGrid(QObject* parent = nullptr) const = 0;
	virtual UnitThreadController* buildSimulationThreads(QObject* parent = nullptr) const = 0;
	virtual SpaceMetric* buildSpaceMetric(QObject* parent = nullptr) const = 0;
	virtual MapCompartment* buildMapCompartment(QObject* parent = nullptr) const = 0;
	virtual CellMap* buildCellMap(QObject* parent = nullptr) const = 0;
	virtual EnergyParticleMap* buildEnergyParticleMap(QObject* parent = nullptr) const = 0;
};

#endif // CONTEXTFACTORY_H
