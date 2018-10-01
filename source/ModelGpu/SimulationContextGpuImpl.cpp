#include "ModelInterface/SymbolTable.h"
#include "ModelInterface/SimulationParameters.h"
#include "ModelInterface/SpaceProperties.h"

#include "WorkerForGpu.h"
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
	SET_CHILD(_metric, metric);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_parameters, parameters);

	auto threadController = new ThreadController;
	SET_CHILD(_threadController, threadController);
	_threadController->init(metric);
}

SpaceProperties * SimulationContextGpuImpl::getSpaceProperties() const
{
	return _metric;
}

IntVector2D SimulationContextGpuImpl::getGridSize() const
{
	return IntVector2D();
}

uint SimulationContextGpuImpl::getMaxThreads() const
{
	return 0;
}

SymbolTable * SimulationContextGpuImpl::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters * SimulationContextGpuImpl::getSimulationParameters() const
{
	return _parameters;
}

CellComputerCompiler * SimulationContextGpuImpl::getCellComputerCompiler() const
{
	return nullptr;
}

void SimulationContextGpuImpl::setSimulationParameters(SimulationParameters * parameters)
{
}

ThreadController * SimulationContextGpuImpl::getGpuThreadController() const
{
	return _threadController;
}

