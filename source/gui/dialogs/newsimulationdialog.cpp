#include "newsimulationdialog.h"
#include "ui_newsimulationdialog.h"

#include "simulationparametersdialog.h"
#include "symboltabledialog.h"

#include "gui/guisettings.h"
#include "global/global.h"
#include "model/metadatamanager.h"

#include <QDebug>

NewSimulationDialog::NewSimulationDialog(QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());

    _simParaDialog = new SimulationParametersDialog();
    _symTblDialog = new SymbolTableDialog();

    //connections
    connect(ui->simulationParametersButton, SIGNAL(clicked()), this, SLOT(simulationParametersButtonClicked()));
    connect(ui->symbolTableButton, SIGNAL(clicked()), this, SLOT(symbolTableButtonClicked()));
    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(okButtonClicked()));
}

NewSimulationDialog::~NewSimulationDialog()
{
    delete _simParaDialog;
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

SymbolTable * NewSimulationDialog::getSymbolTable()
{
	return _symTblDialog->getNewSymbolTableRef();
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{
    _simParaDialog->exec();
}

void NewSimulationDialog::symbolTableButtonClicked ()
{
    _symTblDialog->exec();
}

void NewSimulationDialog::okButtonClicked ()
{
    _simParaDialog->updateSimulationParameters();
}


