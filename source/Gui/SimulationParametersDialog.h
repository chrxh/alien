#pragma once

#include <QDialog>

#include "ModelBasic/SimulationParameters.h"

#include "Gui/Definitions.h"

namespace Ui {
class SimulationParametersDialog;
}

class SimulationParametersDialog
	: public QDialog
{
    Q_OBJECT

public:
    SimulationParametersDialog(SimulationConfig const& config, Serializer* serializer, QWidget *parent = nullptr);
    virtual ~SimulationParametersDialog();

	SimulationParameters* getSimulationParameters ();

private:
	Q_SLOT void okClicked();
    Q_SLOT void updateWidgetsFromSimulationParameters ();
	Q_SLOT void updateSimulationParametersFromWidgets ();

	Q_SLOT void defaultButtonClicked ();
	Q_SLOT void loadButtonClicked ();
	Q_SLOT void saveButtonClicked ();

private:
	void setItem(QString key, int matchPos, int value);
	void setItem(QString key, int matchPos, qreal value);
	int getItemInt(QString key, int matchPos);
	qreal getItemReal(QString key, int matchPos);

	bool saveSimulationParameters(string filename, SimulationParameters* symbolTable);

	Ui::SimulationParametersDialog *ui;
	Serializer* _serializer = nullptr;
    SimulationParameters* _simulationParameters = nullptr;

	SimulationConfig _config;
};
