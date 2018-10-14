#include <QMessageBox>

#include "Gui/Settings.h"
#include "Gui/StringHelper.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "SimulationConfig.h"

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
	ui->gridSizeXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridSizeXKey, Const::GridSizeXDefault)));
	ui->gridSizeYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::GridSizeYKey, Const::GridSizeYDefault)));
	ui->unitSizeXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::UnitSizeXKey, Const::UnitSizeXDefault)));
	ui->unitSizeYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::UnitSizeYKey, Const::UnitSizeYDefault)));
	ui->maxThreadsEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::MaxThreadsKey, Const::MaxThreadsDefault)));
	ui->energyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::InitialEnergyKey, Const::InitialEnergyDefault)));

	updateLabels();

    connect(ui->simulationParametersButton, &QPushButton::clicked, this, &NewSimulationDialog::simulationParametersButtonClicked);
    connect(ui->symbolTableButton, &QPushButton::clicked, this, &NewSimulationDialog::symbolTableButtonClicked);
	connect(ui->gridSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateLabels);
	connect(ui->gridSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateLabels);
	connect(ui->unitSizeXEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateLabels);
	connect(ui->unitSizeYEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateLabels);
	connect(ui->maxThreadsEdit, &QLineEdit::textEdited, this, &NewSimulationDialog::updateLabels);
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

IntVector2D NewSimulationDialog::getUnitSize() const
{
	IntVector2D gridSize = getGridSize();
	IntVector2D universeSize = getUniverseSize();
	return{ universeSize.x / gridSize.x, universeSize.y / gridSize.y };
}

SimulationConfig NewSimulationDialog::createConfig() const
{
	auto config = boost::make_shared<_SimulationConfigCpu>();
	config->maxThreads = getMaxThreads();
	config->universeSize = getUniverseSize();
	config->gridSize = getGridSize();
	config->parameters = getSimulationParameters();
	config->symbolTable = getSymbolTable();

	return config;
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

double NewSimulationDialog::getEnergy () const
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

	SimulationParametersDialog d(createConfig(), _serializer, this);
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

void NewSimulationDialog::updateLabels()
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
	ui->universeSizeXLabel->setText(StringHelper::toString(_universeSize.x));
	ui->universeSizeYLabel->setText(StringHelper::toString(_universeSize.y));
	int limitThreads = getMaxThreads();
	int activeThreads = std::min((gridSizeX / 3) * (gridSizeY / 3), limitThreads);
	int totalThreads = gridSizeX * gridSizeY;
	ui->activeThreadsLabel->setText(StringHelper::toString(activeThreads) + QString(" (active)"));
	ui->totalThreadsLabel->setText(StringHelper::toString(totalThreads) + QString(" (total)"));
}

void NewSimulationDialog::okClicked()
{
	SimulationConfig config = createConfig();
	string errorMsg;
	auto valResult = config->validate(errorMsg);
	if (valResult == _SimulationConfig::ValidationResult::Ok) {
		GuiSettings::setSettingsValue(Const::GridSizeXKey, getGridSize().x);
		GuiSettings::setSettingsValue(Const::GridSizeYKey, getGridSize().y);
		GuiSettings::setSettingsValue(Const::UnitSizeXKey, getUnitSize().x);
		GuiSettings::setSettingsValue(Const::UnitSizeYKey, getUnitSize().y);
		GuiSettings::setSettingsValue(Const::MaxThreadsKey, static_cast<int>(getMaxThreads()));
		GuiSettings::setSettingsValue(Const::InitialEnergyKey, getEnergy());
		accept();
	}
	else if (valResult == _SimulationConfig::ValidationResult::Error) {
		QMessageBox msgBox(QMessageBox::Critical, "error", errorMsg.c_str());
		msgBox.exec();
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}



