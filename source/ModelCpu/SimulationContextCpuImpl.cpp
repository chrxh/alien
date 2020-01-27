#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
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

void SimulationContextCpuImpl::init(SpaceProperties* spaceProp, UnitGrid* grid, UnitThreadController* threads
	, SymbolTable * symbolTable, SimulationParameters const& parameters, CellComputerCompiler* compiler)
{
	_simulationParameters = parameters;

	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init();

	SET_CHILD(_spaceProp, spaceProp);
	SET_CHILD(_grid, grid);
	SET_CHILD(_threads, threads);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_compiler, compiler);
	SET_CHILD(_numberGen, numberGen);

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

SimulationParameters const& SimulationContextCpuImpl::getSimulationParameters() const
{
	return _simulationParameters;
}

NumberGenerator * SimulationContextCpuImpl::getNumberGenerator() const
{
	return _numberGen;
}

int SimulationContextCpuImpl::getTimestep() const
{
    return 0;
}

map<string, int> SimulationContextCpuImpl::getSpecificData() const
{
	ModelCpuData data(_threads->getMaxRunningThreads(), _grid->getSize());
	return data.getData();
}

void SimulationContextCpuImpl::setSimulationParameters(SimulationParameters const& parameters)
{
	_attributeSetter->setSimulationParameters(parameters);
	_simulationParameters = parameters;
}

