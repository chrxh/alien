#ifndef NEWSIMULATIONDIALOG_H
#define NEWSIMULATIONDIALOG_H

#include <QDialog>

namespace Ui {
class NewSimulationDialog;
}

class SimulationParametersDialog;
class SymbolTableDialog;
class NewSimulationDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewSimulationDialog(QWidget *parent = 0);
    ~NewSimulationDialog();

    int getSizeX ();
    int getSizeY ();
    qreal getEnergy ();

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
