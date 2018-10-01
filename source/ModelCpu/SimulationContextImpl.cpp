#include "Base/NumberGenerator.h"

#include "ModelInterface/SimulationParameters.h"
#include "ModelInterface/CellComputerCompiler.h"
#include "ModelInterface/SymbolTable.h"
#include "SpacePropertiesImpl.h"
#include "UnitGrid.h"
#include "UnitThreadController.h"

#include "SimulationAttributeSetter.h"
#include "SimulationContextImpl.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContext(parent)
{
}

SimulationContextImpl::~SimulationContextImpl()
{
	delete _threads;
}

void SimulationContextImpl::init(NumberGenerator* numberGen, SpacePropertiesImpl* metric, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters* parameters, CellComputerCompiler* compiler)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
	SET_CHILD(_compiler, compiler);

	auto attributeSetter = new SimulationAttributeSetter();
	SET_CHILD(_attributeSetter, attributeSetter);

	attributeSetter->init(this);
}

IntVector2D SimulationContextImpl::getGridSize() const
{
	return _grid->getSize();
}

uint SimulationContextImpl::getMaxThreads() const
{
	return _threads->getMaxRunningThreads();
}

SpaceProperties * SimulationContextImpl::getSpaceProperties() const
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

SymbolTable* SimulationContextImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters* SimulationContextImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

NumberGenerator * SimulationContextImpl::getNumberGenerator() const
{
	return _numberGen;
}

CellComputerCompiler * SimulationContextImpl::getCellComputerCompiler() const
{
	return _compiler;
}

void SimulationContextImpl::setSimulationParameters(SimulationParameters * parameters)
{
	_attributeSetter->setSimulationParameters(parameters);
	_simulationParameters = parameters;
}

