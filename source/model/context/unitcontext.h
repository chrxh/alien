#ifndef UNITCONTEXT_H
#define UNITCONTEXT_H

#include "model/UnitContextApi.h"

class UnitContext
	: public UnitContextApi
{
	Q_OBJECT
public:
	UnitContext(QObject* parent = nullptr) : UnitContextApi(parent) {}
	virtual ~UnitContext() = default;
	
	virtual void init(SpaceMetric* metric, CellMap* cellMap, EnergyParticleMap* energyMap, MapCompartment* mapCompartment, SymbolTable* symbolTable
		, SimulationParameters* parameters) = 0;

    virtual SpaceMetric* getSpaceMetric () const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // UNITCONTEXT_H
