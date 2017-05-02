#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include "model/Definitions.h"

class SimulationContext
	: public QObject
{
	Q_OBJECT
public:
	SimulationContext(QObject* parent) : QObject(parent) {}
	virtual ~SimulationContext() {}

	virtual void init(SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters) = 0;

	virtual SpaceMetric* getTopology() const = 0;
	virtual UnitGrid* getSimulationGrid() const = 0;
	virtual UnitThreadController* getSimulationThreads() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // SIMULATIONCONTEXT_H
