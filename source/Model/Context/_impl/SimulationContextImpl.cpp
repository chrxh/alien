#include "Base/NumberGenerator.h"
#include "model/Context/SpaceMetric.h"
#include "model/Context/UnitGrid.h"
#include "model/Context/UnitThreadController.h"
#include "model/Context/SimulationParameters.h"
#include "model/Metadata/SymbolTable.h"

#include "SimulationContextImpl.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContext(parent)
{
}

SimulationContextImpl::~SimulationContextImpl()
{
	delete _threads;
}

void SimulationContextImpl::init(NumberGenerator* numberGen, SpaceMetric* metric, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters* parameters)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
}

SpaceMetricApi * SimulationContextImpl::getSpaceMetric() const
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
