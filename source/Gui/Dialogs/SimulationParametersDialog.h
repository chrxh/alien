#pragma once

#include <QDialog>
#include "Model/Api/SimulationParameters.h"

namespace Ui {
class SimulationParametersDialog;
}

class SimulationParametersDialog : public QDialog
{
    Q_OBJECT

public:
    SimulationParametersDialog(SimulationParameters* parameters, Serializer* serializer, QWidget *parent = nullptr);
    virtual ~SimulationParametersDialog();

	SimulationParameters* getSimulationParameters ();

private:
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

	SimulationParameters* loadSimulationParameters(string filename);
	bool saveSimulationParameters(string filename, SimulationParameters* symbolTable);


	Ui::SimulationParametersDialog *ui;
	Serializer* _serializer = nullptr;
    SimulationParameters* _simulationParameters = nullptr;
};
