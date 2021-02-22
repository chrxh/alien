#pragma once

#include <QDialog>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"

#include "Definitions.h"

namespace Ui {
	class NewSimulationDialog;
}

class SimulationParametersDialog;
class SymbolTableDialog;
class NewSimulationDialog : public QDialog
{
	Q_OBJECT

public:
	NewSimulationDialog(SimulationParameters const& parameters, SymbolTable const* symbols, Serializer* serializer, QWidget* parent = nullptr);
	virtual ~NewSimulationDialog();

	boost::optional<SimulationConfig> getConfig() const;
	boost::optional<double> getEnergy() const;

private:
	SymbolTable* getSymbolTable() const;
	SimulationParameters const& getSimulationParameters() const;

private:
	Q_SLOT void simulationParametersButtonClicked();
	Q_SLOT void symbolTableButtonClicked();
	Q_SLOT void okClicked();

private:
	Ui::NewSimulationDialog *ui;
	Serializer* _serializer = nullptr;

	SimulationParameters _parameters;
	SymbolTable* _symbolTable = nullptr;

	IntVector2D _universeSizeForEngineGpu;
	IntVector2D _gridSize;
};

