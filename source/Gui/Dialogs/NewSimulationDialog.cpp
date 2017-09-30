#include <QDebug>

#include "gui/Settings.h"
#include "Model/Local/UnitContext.h"

#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"

#include "NewSimulationDialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(UnitContext* context, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _localParameters(context->getSimulationParameters()->clone())
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    _symTblDialog = new SymbolTableDialog(context->getSymbolTable());

    //connections
    connect(ui->simulationParametersButton, SIGNAL(clicked()), this, SLOT(simulationParametersButtonClicked()));
    connect(ui->symbolTableButton, SIGNAL(clicked()), this, SLOT(symbolTableButtonClicked()));
}


NewSimulationDialog::~NewSimulationDialog()
{
    delete _symTblDialog;
    delete ui;
}

IntVector2D NewSimulationDialog::getSize ()
{
    bool ok(true);
	return{ ui->sizeXEdit->text().toInt(&ok), ui->sizeYEdit->text().toInt(&ok) };
}

qreal NewSimulationDialog::getEnergy ()
{
    bool ok(true);
    return ui->energyEdit->text().toDouble(&ok);
}

SymbolTable* NewSimulationDialog::getNewSymbolTable()
{
	return _symTblDialog->getNewSymbolTable();
}

SimulationParameters* NewSimulationDialog::getNewSimulationParameters()
{
	return _localParameters;
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{
	SimulationParametersDialog d(_localParameters->clone());
	if (d.exec()) {
		_localParameters = d.getSimulationParameters();
	}
}

void NewSimulationDialog::symbolTableButtonClicked ()
{
    _symTblDialog->exec();
}



