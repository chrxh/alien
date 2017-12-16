#include "Base/NumberGenerator.h"

#include "Model/Api/SimulationParameters.h"
#include "Model/Api/CellComputerCompiler.h"
#include "Model/Api/SymbolTable.h"
#include "Model/Local/SpacePropertiesLocal.h"
#include "Model/Local/UnitGrid.h"
#include "Model/Local/UnitThreadController.h"

#include "SimulationContextImpl.h"

SimulationContextImpl::SimulationContextImpl(QObject * parent)
	: SimulationContextLocal(parent)
{
}

SimulationContextImpl::~SimulationContextImpl()
{
	delete _threads;
}

void SimulationContextImpl::init(NumberGenerator* numberGen, SpacePropertiesLocal* metric, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters* parameters, CellComputerCompiler* compiler)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_metric, metric);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
	SET_CHILD(_compiler, compiler);
}

IntVector2D SimulationContextImpl::getGridSize() const
{
	return _grid->getSize();
}

int SimulationContextImpl::getMaxThreads() const
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

CellComputerCompiler * SimulationContextImpl::getCellComputerCompiler() const
{
	return _compiler;
}
