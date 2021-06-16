#include <QMessageBox>

#include "Gui/Settings.h"
#include "Gui/StringHelper.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "SimulationConfig.h"

#include "NewSimulationDialog.h"
#include "ui_NewSimulationDialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationParameters const& parameters, SymbolTable const* symbols, Serializer* serializer, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _parameters(parameters)
	, _symbolTable(symbols->clone())
	, _serializer(serializer)
{
	_symbolTable->setParent(parent);
	ui->setupUi(this);
	
	ui->energyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::InitialEnergyKey, Const::InitialEnergyDefault)));

    connect(ui->simulationParametersButton, &QPushButton::clicked, this, &NewSimulationDialog::simulationParametersButtonClicked);
    connect(ui->symbolTableButton, &QPushButton::clicked, this, &NewSimulationDialog::symbolTableButtonClicked);
	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &NewSimulationDialog::okClicked);
}


NewSimulationDialog::~NewSimulationDialog()
{
    delete ui;
}

boost::optional<SimulationConfig> NewSimulationDialog::getConfig() const
{
	auto config = boost::make_shared<_SimulationConfig>();
    if (auto const value = ui->computationSettings->getUniverseSize()) {
        config->universeSize = *value;
    }
    else {
        return boost::none;
    }
	config->parameters = getSimulationParameters();
	config->symbolTable = getSymbolTable();
    if (auto const value = ui->computationSettings->getCudaConstants()) {
        config->cudaConstants = *value;
    }
    else {
        return boost::none;
    }
    return config;
}

boost::optional<double> NewSimulationDialog::getEnergy () const
{
    bool ok(true);
	double energy = ui->energyEdit->text().toDouble(&ok);
	if (!ok) {
		return boost::none;
	}
    return energy;
}

SymbolTable* NewSimulationDialog::getSymbolTable() const
{
	return _symbolTable;
}

SimulationParameters const& NewSimulationDialog::getSimulationParameters() const
{
	return _parameters;
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{
	SimulationParametersDialog d(getSimulationParameters(), _serializer, this);
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

void NewSimulationDialog::okClicked()
{
	auto const config = getConfig();
    auto const energy = getEnergy();

    if (!config || !energy) {
        QMessageBox msgBox(QMessageBox::Critical, "Invalid values", Const::ErrorInvalidValues);
        msgBox.exec();
        return;
    }

    ui->computationSettings->saveSettings();
    GuiSettings::setSettingsValue(Const::InitialEnergyKey, *getEnergy());
    accept();
}



