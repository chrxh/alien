#include "MainModel.h"

MainModel::MainModel(QObject * parent) : QObject(parent) {
}

SimulationParameters const* MainModel::getSimulationParameters() const
{
	return _parameters;
}

void MainModel::setSimulationParameters(SimulationParameters const* parameters)
{
	_parameters = parameters;
}

SymbolTable const* MainModel::getSymbolTable() const
{
	return _symbols;
}

void MainModel::setSymbolTable(SymbolTable const* symbols)
{
	_symbols = symbols;
}

void MainModel::setEditMode(optional<bool> value)
{
	_isEditMode = value;
}

optional<bool> MainModel::isEditMode() const
{
	return _isEditMode;
}
