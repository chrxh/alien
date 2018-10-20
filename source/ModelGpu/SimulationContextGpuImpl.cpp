#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"

#include "CudaWorker.h"
#include "ThreadController.h"
#include "SimulationContextGpuImpl.h"

SimulationContextGpuImpl::SimulationContextGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationContext(parent)
{
}

SimulationContextGpuImpl::~SimulationContextGpuImpl()
{
}

void SimulationContextGpuImpl::init(SpaceProperties *metric, SymbolTable *symbolTable, SimulationParameters *parameters)
{
	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init();

	SET_CHILD(_metric, metric);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_parameters, parameters);
	SET_CHILD(_numberGen, numberGen);

	auto threadController = new ThreadController;
	SET_CHILD(_threadController, threadController);
	_threadController->init(metric);
}

SpaceProperties * SimulationContextGpuImpl::getSpaceProperties() const
{
	return _metric;
}

SymbolTable * SimulationContextGpuImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters * SimulationContextGpuImpl::getSimulationParameters() const
{
	return _parameters;
}

NumberGenerator * SimulationContextGpuImpl::getNumberGenerator() const
{
	return _numberGen;
}

map<string, int> SimulationContextGpuImpl::getSpecificData() const
{
	return map<string, int>();
}

void SimulationContextGpuImpl::setSimulationParameters(SimulationParameters * parameters)
{
}

ThreadController * SimulationContextGpuImpl::getGpuThreadController() const
{
	return _threadController;
}

