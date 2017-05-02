#include "SimulationContextImpl.h"
#include "model/context/SpaceMetric.h"
#include "model/context/UnitGrid.h"
#include "model/context/UnitThreadController.h"
#include "model/context/SimulationParameters.h"
#include "model/metadata/SymbolTable.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContext(parent)
{
}

SimulationContextImpl::~SimulationContextImpl()
{
	delete _threads;
}

void SimulationContextImpl::init(SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads, SymbolTable * symbolTable, SimulationParameters* parameters)
{
	if (_metric != metric) {
		delete _metric;
		_metric = metric;
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

SpaceMetric * SimulationContextImpl::getTopology() const
{
	return _metric;
}

UnitGrid * SimulationContextImpl::getSimulationGrid() const
{
	return _grid;
}

UnitThreadController * SimulationContextImpl::getSimulationThreads() const
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
