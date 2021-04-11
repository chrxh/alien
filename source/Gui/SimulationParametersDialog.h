#pragma once

#include <QDialog>

#include "EngineInterface/SimulationParameters.h"

#include "Gui/Definitions.h"

namespace Ui {
class SimulationParametersDialog;
}

class QJsonModel;

class SimulationParametersDialog
	: public QDialog
{
    Q_OBJECT

public:
    SimulationParametersDialog(SimulationParameters const& parameters, Serializer* serializer, QWidget *parent = nullptr);
    virtual ~SimulationParametersDialog();

	SimulationParameters const& getSimulationParameters () const;

private:
	Q_SLOT void okClicked();
	Q_SLOT void defaultButtonClicked ();
	Q_SLOT void loadButtonClicked ();
	Q_SLOT void saveButtonClicked ();

private:
    void updateModelFromSimulationParameters();
    void updateSimulationParametersFromModel();

	bool saveSimulationParameters(string filename);

	Ui::SimulationParametersDialog *ui;
	Serializer* _serializer = nullptr;
    SimulationParameters _simulationParameters;
    QJsonModel* _model = nullptr;
};
