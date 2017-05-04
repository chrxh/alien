#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include "model/SimulationContextHandle.h"

class SimulationContext
	: public SimulationContextHandle
{
	Q_OBJECT
public:
	SimulationContext(QObject* parent = nullptr) : SimulationContextHandle(parent) {}
	virtual ~SimulationContext() = default;

	virtual void init(SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters) = 0;

	virtual void lock() = 0;
	virtual void unlock() = 0;

	virtual SpaceMetric* getSpaceMetric() const = 0;
	virtual UnitGrid* getUnitGrid() const = 0;
	virtual UnitThreadController* getUnitThreadController() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // SIMULATIONCONTEXT_H
