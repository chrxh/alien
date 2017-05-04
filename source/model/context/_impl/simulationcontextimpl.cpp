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
	SET_CHILD(_metric, metric);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
}

void SimulationContextImpl::lock()
{
	_mutex.lock();
}

void SimulationContextImpl::unlock()
{
	_mutex.unlock();
}

SpaceMetric * SimulationContextImpl::getSpaceMetric() const
{
	return _metric;
}

UnitGrid * SimulationContextImpl::getUnitGrid() const
{
	return _grid;
}

UnitThreadController * SimulationContextImpl::getUnitThreadController() const
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
