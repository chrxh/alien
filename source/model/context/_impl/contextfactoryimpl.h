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
	virtual UnitContext* buildSimulationUnitContext(QObject* parent = nullptr) const override;
	virtual Unit* buildSimulationUnit(QObject* parent = nullptr) const override;
	virtual Grid* buildSimulationGrid(QObject* parent = nullptr) const override;
	virtual ThreadController* buildSimulationThreads(QObject* parent = nullptr) const override;
	virtual SpaceMetric* buildSpaceMetric(QObject* parent = nullptr) const override;
	virtual MapCompartment* buildMapCompartment(QObject* parent = nullptr) const override;
	virtual CellMap* buildCellMap(QObject* parent = nullptr) const override;
	virtual EnergyParticleMap* buildEnergyParticleMap(QObject* parent = nullptr) const override;
};

#endif // CONTEXTFACTORYIMPL_H
