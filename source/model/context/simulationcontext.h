#ifndef SIMULATIONCONTEXT_H
#define SIMULATIONCONTEXT_H

#include "model/definitions.h"

class SimulationContext
	: public QObject
{
	Q_OBJECT
public:
	SimulationContext(QObject* parent) : QObject(parent) {}
	virtual ~SimulationContext() {}

	virtual void init(SpaceMetric* metric, SimulationGrid* grid, ThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters) = 0;

	virtual SpaceMetric* getTopology() const = 0;
	virtual SimulationGrid* getSimulationGrid() const = 0;
	virtual ThreadController* getSimulationThreads() const = 0;
	virtual SymbolTable* getSymbolTable() const = 0;
	virtual SimulationParameters* getSimulationParameters() const = 0;
};

#endif // SIMULATIONCONTEXT_H
