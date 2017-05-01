#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include <QMutex>
#include "model/context/simulationcontext.h"

class SimulationContextImpl
	: public SimulationContext
{
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl() {}

	virtual void init(Topology* topology, SimulationGrid* grid, SimulationThreads* threads, SymbolTable * symbolTable, SimulationParameters* parameters) override;

	virtual Topology* getTopology() const override;
	virtual SimulationGrid* getSimulationGrid() const override;
	virtual SimulationThreads* getSimulationThreads() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	Topology* _topology = nullptr;
	SimulationGrid* _grid = nullptr;
	SimulationThreads* _threads = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
