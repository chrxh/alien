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

	virtual void lock();
	virtual void unlock();

	virtual Topology* getTopology() const override;
	virtual SimulationGrid* getSimulationGrid() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	QMutex _mutex;
	Topology* _topology = nullptr;
	SimulationGrid* _grid = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
