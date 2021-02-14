#include "MainModel.h"

MainModel::MainModel(QObject * parent) : QObject(parent) {
}

SimulationParameters MainModel::getSimulationParameters() const
{
	return _simulationParameters;
}

void MainModel::setSimulationParameters(SimulationParameters const& parameters)
{
	_simulationParameters = parameters;
}

ExecutionParameters MainModel::getExecutionParameters() const
{
    return _executionParameters;
}

void MainModel::setExecutionParameters(ExecutionParameters const & parameters)
{
    _executionParameters = parameters;
}

SymbolTable * MainModel::getSymbolTable() const
{
	return _symbols;
}

void MainModel::setSymbolTable(SymbolTable * symbols)
{
	_symbols = symbols;
}
