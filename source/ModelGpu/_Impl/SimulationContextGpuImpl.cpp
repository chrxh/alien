#include "Model/Local/SymbolTable.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Api/SpaceMetric.h"

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

void SimulationContextGpuImpl::init(SpaceMetric *metric, SymbolTable *symbolTable, SimulationParameters *parameters)
{
	SET_CHILD(_metric, metric);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_parameters, parameters);

	auto threadController = new ThreadController;
	SET_CHILD(_threadController, threadController);
	_threadController->init(metric);
}

SpaceMetric * SimulationContextGpuImpl::getSpaceMetric() const
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

ThreadController * SimulationContextGpuImpl::getGpuThreadController() const
{
	return _threadController;
}

