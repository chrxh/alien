#ifndef NEWSIMULATIONDIALOG_H
#define NEWSIMULATIONDIALOG_H

#include <QDialog>

#include "Model/Api/Definitions.h"
#include "Model/Api/SimulationParameters.h"

namespace Ui {
class NewSimulationDialog;
}

class SimulationParametersDialog;
class SymbolTableDialog;
class NewSimulationDialog : public QDialog
{
    Q_OBJECT

public:
	NewSimulationDialog(SimulationParameters* parameters, SymbolTable* symbols, QWidget* parent = 0);
    virtual ~NewSimulationDialog();

    IntVector2D getSize();
    qreal getEnergy();
	SymbolTable* getNewSymbolTable();
	SimulationParameters* getNewSimulationParameters();

private Q_SLOTS:
    void simulationParametersButtonClicked ();
    void symbolTableButtonClicked ();

private:
    Ui::NewSimulationDialog *ui;
    SymbolTableDialog* _symTblDialog;

	SimulationParameters* _localParameters;
};

#endif // NEWSIMULATIONDIALOG_H
