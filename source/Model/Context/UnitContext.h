#ifndef UNITCONTEXT_H
#define UNITCONTEXT_H

#include "Model/Definitions.h"

class UnitContext
	: public QObject
{
	Q_OBJECT
public:
	UnitContext(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~UnitContext() = default;
	
	virtual void init(NumberGenerator* numberGen, SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap
		, MapCompartment* mapCompartment, SymbolTable* symbolTable, SimulationParameters* parameters) = 0;

	virtual NumberGenerator* getNumberGenerator() const = 0;
    virtual SpaceMetric* getSpaceMetric() const = 0;
	virtual CellMap* getCellMap() const = 0;
	virtual EnergyParticleMap* getEnergyParticleMap() const = 0;
	virtual MapCompartment* getMapCompartment() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;

	virtual uint64_t getTimestamp() const = 0;
	virtual void incTimestamp() = 0;

	virtual QList<CellCluster*>& getClustersRef() = 0;
	virtual QList<EnergyParticle*>& getEnergyParticlesRef() = 0;
};

#endif // UNITCONTEXT_H
