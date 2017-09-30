#pragma once

#include "Model/Local/ContextFactory.h"

class ContextFactoryImpl
	: public ContextFactory
{
public:
	virtual ~ContextFactoryImpl() {}

	virtual SimulationContextLocal* buildSimulationContext() const override;
	virtual UnitContext* buildSimulationUnitContext() const override;
	virtual Unit* buildSimulationUnit() const override;
	virtual UnitGrid* buildSimulationGrid() const override;
	virtual UnitThreadController* buildSimulationThreads() const override;
	virtual SpaceMetricLocal* buildSpaceMetric() const override;
	virtual MapCompartment* buildMapCompartment() const override;
	virtual CellMap* buildCellMap() const override;
	virtual ParticleMap* buildEnergyParticleMap() const override;
};

