#include "simulationcontextimpl.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContext(parent)
{
}

void SimulationContextImpl::lock()
{
	_mutex.lock();
}

void SimulationContextImpl::unlock()
{
	_mutex.unlock();
}

Topology * SimulationContextImpl::getTopology() const
{
	return _topology;
}

SimulationGrid * SimulationContextImpl::getSimulationGrid() const
{
	return _grid;
}

SymbolTable * SimulationContextImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters * SimulationContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}
