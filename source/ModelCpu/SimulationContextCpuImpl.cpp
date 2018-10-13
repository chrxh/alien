#include "Base/NumberGenerator.h"

#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/CellComputerCompiler.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SpaceProperties.h"

#include "UnitGrid.h"
#include "UnitThreadController.h"
#include "ModelCpuData.h"

#include "SimulationAttributeSetter.h"
#include "SimulationContextCpuImpl.h"

SimulationContextCpuImpl::SimulationContextCpuImpl(QObject * parent)
	: SimulationContext(parent)
{
}

SimulationContextCpuImpl::~SimulationContextCpuImpl()
{
	delete _threads;
}

void SimulationContextCpuImpl::init(NumberGenerator* numberGen, SpaceProperties* spaceProp, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters* parameters, CellComputerCompiler* compiler)
{
	SET_CHILD(_numberGen, numberGen);
	SET_CHILD(_spaceProp, spaceProp);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_simulationParameters, parameters);
	SET_CHILD(_compiler, compiler);

	auto attributeSetter = new SimulationAttributeSetter();
	SET_CHILD(_attributeSetter, attributeSetter);

	attributeSetter->init(this);
}

SpaceProperties * SimulationContextCpuImpl::getSpaceProperties() const
{
	return _spaceProp;
}

UnitGrid * SimulationContextCpuImpl::getUnitGrid() const
{
	return _grid;
}

UnitThreadController * SimulationContextCpuImpl::getUnitThreadController() const
{
	return _threads;
}

SymbolTable* SimulationContextCpuImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters* SimulationContextCpuImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

NumberGenerator * SimulationContextCpuImpl::getNumberGenerator() const
{
	return _numberGen;
}

map<string, int> SimulationContextCpuImpl::getSpecificData() const
{
	ModelCpuData data(_threads->getMaxRunningThreads(), _grid->getSize());
	return data.getData();
}

CellComputerCompiler * SimulationContextCpuImpl::getCellComputerCompiler() const
{
	return _compiler;
}

void SimulationContextCpuImpl::setSimulationParameters(SimulationParameters * parameters)
{
	_attributeSetter->setSimulationParameters(parameters);
	_simulationParameters = parameters;
}

