#pragma once

#include "ModelBasic/Definitions.h"
#include "Definitions.h"

class ContextFactory
{
public:
	virtual ~ContextFactory() = default;

	virtual SimulationContextCpuImpl* buildSimulationContext() const = 0;
	virtual UnitContext* buildSimulationUnitContext() const = 0;
	virtual Unit* buildSimulationUnit() const = 0;
	virtual UnitGrid* buildSimulationGrid() const = 0;
	virtual UnitThreadController* buildSimulationThreads() const = 0;
	virtual MapCompartment* buildMapCompartment() const = 0;
	virtual CellMap* buildCellMap() const = 0;
	virtual ParticleMap* buildEnergyParticleMap() const = 0;
};
