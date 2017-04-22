#include <QDebug>

#include "gui/guisettings.h"
#include "global/numbergenerator.h"
#include "model/simulationcontext.h"

#include "simulationparametersdialog.h"
#include "symboltabledialog.h"

#include "newsimulationdialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationContext* context, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());

    _simParaDialog = new SimulationParametersDialog(context->getSimulationParameters());
    _symTblDialog = new SymbolTableDialog(context->getSymbolTable());

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

SymbolTable const& NewSimulationDialog::getNewSymbolTableRef()
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


