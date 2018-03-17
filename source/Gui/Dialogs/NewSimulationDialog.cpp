#include <QMessageBox>

#include "Model/Local/UnitContext.h"

#include "Gui/Settings.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "SimulationParametersValidation.h"

#include "NewSimulationDialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationParameters const* parameters, SymbolTable const* symbols, Serializer* serializer, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _parameters(parameters->clone())
	, _symbolTable(symbols->clone())
	, _serializer(serializer)
{
	_parameters->setParent(parent);
	_symbolTable->setParent(parent);
	ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());

	updateUniverseSize();

    connect(ui->simulationParametersButton, &QPushButton::clicked, this, &NewSimulationDialog::simulationParametersButtonClicked);
    connect(ui->symbolTableButton, &QPushButton::clicked, this, &NewSimulationDialog::symbolTableButtonClicked);
	connect(ui->gridSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->gridSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->unitSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->unitSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateUniverseSize);
	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewSimulationDialog::okClicked);
}


NewSimulationDialog::~NewSimulationDialog()
{
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
	return _symbolTable;
}

SimulationParameters* NewSimulationDialog::getSimulationParameters() const
{
	return _parameters;
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{
	SimulationParametersDialog d(_universeSize, _gridSize, _parameters->clone(), _serializer, this);
	if (d.exec()) {
		_parameters = d.getSimulationParameters();
	}
}

void NewSimulationDialog::symbolTableButtonClicked ()
{
	SymbolTableDialog d(_symbolTable->clone(), _serializer, this);
	if (d.exec()) {
		_symbolTable = d.getSymbolTable();
	}
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

void NewSimulationDialog::okClicked()
{
	if (SimulationParametersValidation::validate(getUniverseSize(), getGridSize(), getSimulationParameters())) {
		accept();
	}
}



