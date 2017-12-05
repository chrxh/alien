#include "Gui/Settings.h"
#include "Model/Local/UnitContext.h"

#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"

#include "NewSimulationDialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationParameters* parameters, SymbolTable* symbols, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _localParameters(parameters->clone())
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

    _symTblDialog = new SymbolTableDialog(symbols);

	updateUniverseSize();

    connect(ui->simulationParametersButton, &QPushButton::clicked, this, &NewSimulationDialog::simulationParametersButtonClicked);
    connect(ui->symbolTableButton, &QPushButton::clicked, this, &NewSimulationDialog::symbolTableButtonClicked);
	connect(ui->gridSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->gridSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->unitSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->unitSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
}


NewSimulationDialog::~NewSimulationDialog()
{
    delete _symTblDialog;
    delete ui;
}

IntVector2D NewSimulationDialog::getUniverseSize () const
{
	return _universeSize;
}

IntVector2D NewSimulationDialog::getGridSize() const
{
	return _gridSize;
}

uint NewSimulationDialog::getMaxThreads() const
{
	bool ok(true);
	uint maxThreads = ui->maxThreadsEdit->text().toUInt(&ok);
	if (!ok) {
		return 0;
	}
	return maxThreads;
}

qreal NewSimulationDialog::getEnergy () const
{
    bool ok(true);
	double energy = ui->energyEdit->text().toDouble(&ok);
	if (!ok) {
		return 0.0;
	}
    return energy;
}

SymbolTable* NewSimulationDialog::getSymbolTable() const
{
	return _symTblDialog->getNewSymbolTable();
}

SimulationParameters* NewSimulationDialog::getSimulationParameters() const
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

void NewSimulationDialog::updateUniverseSize()
{
	bool ok = false;
	int gridSizeX = ui->gridSizeXEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int gridSizeY = ui->gridSizeYEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int unitSizeX = ui->unitSizeXEdit->text().toUInt(&ok);
	if (!ok) { return; }

	int unitSizeY = ui->unitSizeYEdit->text().toUInt(&ok);
	if (!ok) { return; }

	_universeSize = { gridSizeX * unitSizeX, gridSizeY * unitSizeY };
	_gridSize = { gridSizeX, gridSizeY };
	ui->universeSizeXLabel->setText(QString::fromStdString(std::to_string(_universeSize.x)));
	ui->universeSizeYLabel->setText(QString::fromStdString(std::to_string(_universeSize.y)));
}



