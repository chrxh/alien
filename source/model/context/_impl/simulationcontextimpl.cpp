#include "simulationcontextimpl.h"
#include "model/context/topology.h"
#include "model/context/simulationgrid.h"
#include "model/context/simulationthreads.h"
#include "model/context/simulationparameters.h"
#include "model/metadata/symboltable.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContext(parent)
{
}

void SimulationContextImpl::init(Topology* topology, SimulationGrid* grid, SimulationThreads* threads, SymbolTable * symbolTable, SimulationParameters* parameters)
{
	if (_topology != topology) {
		delete _topology;
		_topology = topology;
	}
	if (_grid != grid) {
		delete _grid;
		_grid = grid;
	}
	if (_threads != threads) {
		delete _threads;
		_threads = threads;
	}
	if (_symbolTable != symbolTable) {
		delete _symbolTable;
		_symbolTable = symbolTable;
	}
	if (_simulationParameters != parameters) {
		delete _simulationParameters;
		_simulationParameters = parameters;
	}
}

Topology * SimulationContextImpl::getTopology() const
{
	return _topology;
}

SimulationGrid * SimulationContextImpl::getSimulationGrid() const
{
	return _grid;
}

SimulationThreads * SimulationContextImpl::getSimulationThreads() const
{
	return _threads;
}

SymbolTable * SimulationContextImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters * SimulationContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}
