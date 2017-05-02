#ifndef SIMULATIONCONTEXTIMPL_H
#define SIMULATIONCONTEXTIMPL_H

#include <QMutex>
#include "model/context/SimulationContext.h"

class SimulationContextImpl
	: public SimulationContext
{
	Q_OBJECT
public:
	SimulationContextImpl(QObject* parent = nullptr);
	virtual ~SimulationContextImpl();

	virtual void init(SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters) override;

	virtual SpaceMetric* getTopology() const override;
	virtual UnitGrid* getSimulationGrid() const override;
	virtual UnitThreadController* getSimulationThreads() const override;
	virtual SymbolTable* getSymbolTable() const override;
	virtual SimulationParameters* getSimulationParameters() const override;

private:
	SpaceMetric* _metric = nullptr;
	UnitGrid* _grid = nullptr;
	UnitThreadController* _threads = nullptr;
	SymbolTable* _symbolTable = nullptr;
	SimulationParameters* _simulationParameters = nullptr;
};

#endif // SIMULATIONCONTEXTIMPL_H
