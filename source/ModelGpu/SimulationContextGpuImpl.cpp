#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"

#include "CudaWorker.h"
#include "CudaController.h"
#include "SimulationContextGpuImpl.h"

SimulationContextGpuImpl::SimulationContextGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationContext(parent)
{
}

void SimulationContextGpuImpl::init(SpaceProperties *space, SymbolTable *symbolTable, SimulationParameters const& parameters)
{
	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(1323781, 1);

	SET_CHILD(_metric, space);
	SET_CHILD(_symbolTable, symbolTable);
	_parameters = parameters;
	SET_CHILD(_numberGen, numberGen);

	auto cudaController = new CudaController;
	SET_CHILD(_cudaController, cudaController);
	_cudaController->init(space, parameters);
}

SpaceProperties * SimulationContextGpuImpl::getSpaceProperties() const
{
	return _metric;
}

SymbolTable * SimulationContextGpuImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters const& SimulationContextGpuImpl::getSimulationParameters() const
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

void SimulationContextGpuImpl::setSimulationParameters(SimulationParameters const& parameters)
{
	_parameters = parameters;
	_cudaController->setSimulationParameters(parameters);
}

CudaController * SimulationContextGpuImpl::getCudaController() const
{
	return _cudaController;
}

