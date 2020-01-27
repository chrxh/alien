#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"

#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SpaceProperties.h"

#include "CudaWorker.h"
#include "CudaController.h"
#include "SimulationContextGpuImpl.h"
#include "ModelGpuData.h"
#include "CudaConstants.h"

SimulationContextGpuImpl::SimulationContextGpuImpl(QObject* parent /*= nullptr*/)
	: SimulationContext(parent)
{
}

void SimulationContextGpuImpl::init(
    SpaceProperties* space,
    int timestep,
    SymbolTable* symbolTable,
    SimulationParameters const& parameters,
    ModelGpuData const& specificData)
{
	auto factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	auto numberGen = factory->buildRandomNumberGenerator();
	numberGen->init(1323781, 1);

	SET_CHILD(_metric, space);
	SET_CHILD(_symbolTable, symbolTable);
	_parameters = parameters;
    _specificData = specificData;
	SET_CHILD(_numberGen, numberGen);

	auto cudaController = new CudaController;
	SET_CHILD(_cudaController, cudaController);

	_cudaController->init(space, timestep, parameters, specificData.getCudaConstants());
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
	return _specificData.getData();
}

int SimulationContextGpuImpl::getTimestep() const
{
    return _cudaController->getCudaWorker()->getTimestep();
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


