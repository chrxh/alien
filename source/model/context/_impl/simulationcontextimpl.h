#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include <QMutex>
#include "model/context/simulationcontext.h"

class SimulationContextImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();

	virtual void init(SpaceMetric* metric, Grid* grid, ThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters) override;

	virtual SpaceMetric* getTopology() const override;
	virtual Grid* getSimulationGrid() const override;
	virtual ThreadController* getSimulationThreads() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	SpaceMetric* _metric = nullptr;
	Grid* _grid = nullptr;
	ThreadController* _threads = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
