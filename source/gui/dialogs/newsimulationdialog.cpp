#include <QDebug>

#include "gui/guisettings.h"
#include "global/numbergenerator.h"
#include "model/simulationunitcontext.h"

#include "simulationparametersdialog.h"
#include "symboltabledialog.h"

#include "newsimulationdialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationUnitContext* context, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _localParameters(*context->getSimulationParameters())
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());

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

SymbolTable const& NewSimulationDialog::getNewSymbolTableRef()
{
	return _symTblDialog->getNewSymbolTableRef();
}

SimulationParameters NewSimulationDialog::getNewSimulationParameters()
{
	return _localParameters;
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{
	SimulationParametersDialog d(&_localParameters);
	if (d.exec()) {
		_localParameters = d.getSimulationParameters();
	}
}

void NewSimulationDialog::symbolTableButtonClicked ()
{
    _symTblDialog->exec();
}



