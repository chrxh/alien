#include "MainModel.h"

MainModel::MainModel(QObject * parent) : QObject(parent) {
}

SimulationParameters const& MainModel::getSimulationParameters() const
{
	return _parameters;
}

void MainModel::setSimulationParameters(SimulationParameters const& parameters)
{
	_parameters = parameters;
}

SymbolTable * MainModel::getSymbolTable() const
{
	return _symbols;
}

void MainModel::setSymbolTable(SymbolTable * symbols)
{
	_symbols = symbols;
}

int MainModel::getTPS() const
{
	return _tps;
}

void MainModel::setTPS(int value)
{
	_tps = value;
}
