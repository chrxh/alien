#include "Base/NumberGenerator.h"
#include "Model/Context/SpaceMetricLocal.h"
#include "Model/Context/UnitGrid.h"
#include "Model/Context/UnitThreadController.h"
#include "Model/SimulationParameters.h"
#include "Model/Metadata/SymbolTable.h"

#include "SimulationContextImpl.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContextLocal(parent)
{
}

SimulationContextImpl::~SimulationContextImpl()
{
	delete _threads;
}

void SimulationContextImpl::init(NumberGenerator* numberGen, SpaceMetricLocal* metric, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters* parameters)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
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

NumberGenerator * SimulationContextImpl::getNumberGenerator() const
{
	return _numberGen;
}
