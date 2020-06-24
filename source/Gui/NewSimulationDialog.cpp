#include <QMessageBox>

#include "Gui/Settings.h"
#include "Gui/StringHelper.h"
#include "SimulationParametersDialog.h"
#include "SymbolTableDialog.h"
#include "SimulationConfig.h"

#include "NewSimulationDialog.h"
#include "ui_newsimulationdialog.h"

NewSimulationDialog::NewSimulationDialog(SimulationParameters const& parameters, SymbolTable const* symbols, Serializer* serializer, QWidget *parent)
	: QDialog(parent)
	, ui(new Ui::NewSimulationDialog)
	, _parameters(parameters)
	, _symbolTable(symbols->clone())
	, _serializer(serializer)
{
	_symbolTable->setParent(parent);
	ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
	
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

namespace
{
    uint getUIntOrZero(QString const& string)
    {
        bool ok(true);
        auto const value = string.toUInt(&ok);
        if (!ok) {
            return 0;
        }
        return value;
    }
}

SimulationConfig NewSimulationDialog::getConfig() const
{
	auto config = boost::make_shared<_SimulationConfigGpu>();
	config->universeSize = ui->computationSettings->getUniverseSize();
	config->parameters = getSimulationParameters();
	config->symbolTable = getSymbolTable();
    config->numBlocks = ui->computationSettings->getNumBlocks();
    config->numThreadsPerBlock = ui->computationSettings->getNumThreadsPerBlock();
    config->maxClusters = ui->computationSettings->getMaxClusters();
    config->maxCells = ui->computationSettings->getMaxCells();
    config->maxTokens = ui->computationSettings->getMaxTokens();
    config->maxParticles = ui->computationSettings->getMaxParticles();
    config->dynamicMemorySize = ui->computationSettings->getDynamicMemorySize();
    config->metadataDynamicMemorySize = ui->computationSettings->getMetadataDynamicMemorySize();
    return config;
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

SimulationParameters const& NewSimulationDialog::getSimulationParameters() const
{
	return _parameters;
}

void NewSimulationDialog::simulationParametersButtonClicked ()
{

	SimulationParametersDialog d(getConfig(), _serializer, this);
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
	SimulationConfig config = getConfig();
	string errorMsg;
	auto valResult = config->validate(errorMsg);
	if (valResult == _SimulationConfig::ValidationResult::Ok) {
        ui->computationSettings->saveSettings();
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



