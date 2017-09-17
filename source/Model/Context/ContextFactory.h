#ifndef CONTEXTFACTORY_H
#define CONTEXTFACTORY_H

#include "Model/Definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() = default;

	virtual SimulationContext* buildSimulationContext() const = 0;
	virtual UnitContext* buildSimulationUnitContext() const = 0;
	virtual Unit* buildSimulationUnit() const = 0;
	virtual UnitGrid* buildSimulationGrid() const = 0;
	virtual UnitThreadController* buildSimulationThreads() const = 0;
	virtual SpaceMetric* buildSpaceMetric() const = 0;
	virtual MapCompartment* buildMapCompartment() const = 0;
	virtual CellMap* buildCellMap() const = 0;
	virtual ParticleMap* buildEnergyParticleMap() const = 0;
};

#endif // CONTEXTFACTORY_H
