#include "Model/SpaceMetricApi.h"
#include "Model/Metadata/SymbolTable.h"
#include "Model/Context/SimulationParameters.h"
#include "Model/SpaceMetricApi.h"

#include "SimulationContextGpuImpl.h"

void SimulationContextGpuImpl::init(SpaceMetricApi *metric, SymbolTable *symbolTable, SimulationParameters *parameters)
{
	SET_CHILD(_metric, metric);
	SET_CHILD(_symbolTable, symbolTable);
	SET_CHILD(_parameters, parameters);
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
