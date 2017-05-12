#ifndef CONTEXTFACTORYIMPL_H
#define CONTEXTFACTORYIMPL_H

#include "model/Context/ContextFactory.h"

class ContextFactoryImpl
	: public ContextFactory
{
public:
	ContextFactoryImpl();
	virtual ~ContextFactoryImpl() {}

	virtual SimulationContext* buildSimulationContext() const override;
	virtual UnitContext* buildSimulationUnitContext() const override;
	virtual Unit* buildSimulationUnit() const override;
	virtual UnitGrid* buildSimulationGrid() const override;
	virtual UnitThreadController* buildSimulationThreads() const override;
	virtual SpaceMetric* buildSpaceMetric() const override;
	virtual MapCompartment* buildMapCompartment() const override;
	virtual CellMap* buildCellMap() const override;
	virtual EnergyParticleMap* buildEnergyParticleMap() const override;
};

#endif // CONTEXTFACTORYIMPL_H
