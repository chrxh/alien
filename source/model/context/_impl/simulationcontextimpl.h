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

	virtual void init(SpaceMetric* metric, SimulationGrid* grid, SimulationThreads* threads, SymbolTable * symbolTable, SimulationParameters* parameters) override;

	virtual SpaceMetric* getTopology() const override;
	virtual SimulationGrid* getSimulationGrid() const override;
	virtual SimulationThreads* getSimulationThreads() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	SpaceMetric* _metric = nullptr;
	SimulationGrid* _grid = nullptr;
	SimulationThreads* _threads = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
