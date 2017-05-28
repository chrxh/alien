#include "SimulationContextGpuImpl.h"

void SimulationContextGpuImpl::init(SpaceMetricApi *metric, SymbolTable *symbolTable, SimulationParameters *parameters)
{
	_metric = metric;
	_symbolTable = symbolTable;
	_parameters = parameters;
}

SpaceMetricApi * SimulationContextGpuImpl::getSpaceMetric() const
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
