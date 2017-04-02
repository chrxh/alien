#ifndef NEWSIMULATIONDIALOG_H
#define NEWSIMULATIONDIALOG_H

#include <QDialog>

#include "model/definitions.h"

namespace Ui {
class NewSimulationDialog;
}

class SimulationParametersDialog;
class SymbolTableDialog;
class NewSimulationDialog : public QDialog
{
    Q_OBJECT

public:
	NewSimulationDialog(SimulationContext* context, QWidget* parent = 0);
    ~NewSimulationDialog();

    IntVector2D getSize();
    qreal getEnergy();
	SymbolTable const& getNewSymbolTableRef();

private slots:
    void simulationParametersButtonClicked ();
    void symbolTableButtonClicked ();
    void okButtonClicked ();

private:
    Ui::NewSimulationDialog *ui;
    SimulationParametersDialog* _simParaDialog;
    SymbolTableDialog* _symTblDialog;
};

#endif // NEWSIMULATIONDIALOG_H
