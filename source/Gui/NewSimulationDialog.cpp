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
	ui->gridSizeXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::CpuGridSizeXKey, Const::CpuGridSizeXDefault)));
	ui->gridSizeYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::CpuGridSizeYKey, Const::CpuGridSizeYDefault)));
	ui->unitSizeXEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::CpuUnitSizeXKey, Const::CpuUnitSizeXDefault)));
	ui->unitSizeYEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::CpuUnitSizeYKey, Const::CpuUnitSizeYDefault)));
	ui->maxThreadsEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::CpuMaxThreadsKey, Const::CpuMaxThreadsDefault)));

    ui->gpuUniverseSizeXEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuUniverseSizeXKey, Const::GpuUniverseSizeXDefault)));
    ui->gpuUniverseSizeYEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuUniverseSizeYKey, Const::GpuUniverseSizeYDefault)));
    ui->gpuNumBlocksEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuNumBlocksKey, Const::GpuNumBlocksDefault)));
    ui->gpuNumThreadsPerBlockEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuNumThreadsPerBlockKey, Const::GpuNumThreadsPerBlockDefault)));
    ui->gpuMaxClustersEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxClustersKey, Const::GpuMaxClustersDefault)));
    ui->gpuMaxCellsEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxCellsKey, Const::GpuMaxCellsDefault)));
    ui->gpuMaxTokensEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxTokensKey, Const::GpuMaxTokensDefault)));
    ui->gpuMaxParticlesEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuMaxParticlesKey, Const::GpuMaxParticlesDefault)));
    ui->gpuDynamicMemorySizeEdit->setText(StringHelper::toString(
        GuiSettings::getSettingsValue(Const::GpuDynamicMemorySizeKey, Const::GpuDynamicMemorySizeDefault)));

	ui->energyEdit->setText(StringHelper::toString(
		GuiSettings::getSettingsValue(Const::InitialEnergyKey, Const::InitialEnergyDefault)));

    auto const modelType = static_cast<ModelComputationType>(
        GuiSettings::getSettingsValue(Const::ModelComputationTypeKey, static_cast<int>(Const::ModelComputationTypeDefault)));
    if (ModelComputationType::Cpu == modelType) {
        ui->cpuDeviceRadioButton->setChecked(true);
        ui->gpuFrame->hide();
    }
    else if (ModelComputationType::Gpu == modelType) {
        ui->gpuDeviceRadioButton->setChecked(true);
        ui->cpuFrame->hide();
    }
    else {
        THROW_NOT_IMPLEMENTED();
    }

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

IntVector2D NewSimulationDialog::getUniverseSizeForModelCpu () const
{
	return _universeSizeForModelCpu;
}

IntVector2D NewSimulationDialog::getUniverseSizeForModelGpu() const
{
	IntVector2D result;
	bool ok;
	result.x = ui->gpuUniverseSizeXEdit->text().toUInt(&ok);
	if (!ok) { result.x = 1; }

	result.y = ui->gpuUniverseSizeYEdit->text().toUInt(&ok);
	if (!ok) { result.y = 1; }

	return result;
}

IntVector2D NewSimulationDialog::getUnitSize() const
{
	IntVector2D gridSize = getGridSize();
	IntVector2D universeSize = getUniverseSizeForModelCpu();
	return{ universeSize.x / gridSize.x, universeSize.y / gridSize.y };
}

ModelComputationType NewSimulationDialog::getModelType() const
{
    if (ui->cpuDeviceRadioButton->isChecked()) {
        return ModelComputationType::Cpu;
    }
    else if (ui->gpuDeviceRadioButton->isChecked()) {
        return ModelComputationType::Gpu;
    }
    THROW_NOT_IMPLEMENTED();
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

uint NewSimulationDialog::getNumBlocks() const
{
    return getUIntOrZero(ui->gpuNumBlocksEdit->text());
}

uint NewSimulationDialog::getNumThreadsPerBlock() const
{
    return getUIntOrZero(ui->gpuNumThreadsPerBlockEdit->text());
}

uint NewSimulationDialog::getMaxClusters() const
{
    return getUIntOrZero(ui->gpuMaxClustersEdit->text());
}

uint NewSimulationDialog::getMaxCells() const
{
    return getUIntOrZero(ui->gpuMaxCellsEdit->text());
}

uint NewSimulationDialog::getMaxTokens() const
{
    return getUIntOrZero(ui->gpuMaxTokensEdit->text());
}

uint NewSimulationDialog::getMaxParticles() const
{
    return getUIntOrZero(ui->gpuMaxParticlesEdit->text());
}

uint NewSimulationDialog::getDynamicMemorySize() const
{
    return getUIntOrZero(ui->gpuDynamicMemorySizeEdit->text());
}

SimulationConfig NewSimulationDialog::getConfig() const
{
    auto const modelType = getModelType();
	if (ModelComputationType::Cpu == modelType) {
		auto config = boost::make_shared<_SimulationConfigCpu>();
		config->maxThreads = getMaxThreads();
		config->universeSize = getUniverseSizeForModelCpu();
		config->gridSize = getGridSize();
		config->parameters = getSimulationParameters();
		config->symbolTable = getSymbolTable();
		return config;
	}
	else if (ModelComputationType::Gpu == modelType) {
		auto config = boost::make_shared<_SimulationConfigGpu>();
		config->universeSize = getUniverseSizeForModelGpu();
		config->parameters = getSimulationParameters();
		config->symbolTable = getSymbolTable();
        config->numBlocks = getNumBlocks();
        config->numThreadsPerBlock = getNumThreadsPerBlock();
        config->maxClusters = getMaxClusters();
        config->maxCells = getMaxCells();
        config->maxTokens = getMaxTokens();
        config->maxParticles = getMaxParticles();
        config->dynamicMemorySize = getDynamicMemorySize();
        return config;
	}
	else {
		THROW_NOT_IMPLEMENTED();
	}
}

IntVector2D NewSimulationDialog::getGridSize() const
{
	return _gridSize;
}

uint NewSimulationDialog::getMaxThreads() const
{
    return getUIntOrZero(ui->maxThreadsEdit->text());
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

	_universeSizeForModelCpu = { gridSizeX * unitSizeX, gridSizeY * unitSizeY };
	_gridSize = { gridSizeX, gridSizeY };
	ui->universeSizeXLabel->setText(StringHelper::toString(_universeSizeForModelCpu.x));
	ui->universeSizeYLabel->setText(StringHelper::toString(_universeSizeForModelCpu.y));
	int limitThreads = getMaxThreads();
	int activeThreads = std::min((gridSizeX / 3) * (gridSizeY / 3), limitThreads);
	int totalThreads = gridSizeX * gridSizeY;
	ui->activeThreadsLabel->setText(StringHelper::toString(activeThreads) + QString(" (active)"));
	ui->totalThreadsLabel->setText(StringHelper::toString(totalThreads) + QString(" (total)"));
}

void NewSimulationDialog::okClicked()
{
	SimulationConfig config = getConfig();
	string errorMsg;
	auto valResult = config->validate(errorMsg);
	if (valResult == _SimulationConfig::ValidationResult::Ok) {
		GuiSettings::setSettingsValue(Const::CpuGridSizeXKey, getGridSize().x);
		GuiSettings::setSettingsValue(Const::CpuGridSizeYKey, getGridSize().y);
		GuiSettings::setSettingsValue(Const::CpuUnitSizeXKey, getUnitSize().x);
		GuiSettings::setSettingsValue(Const::CpuUnitSizeYKey, getUnitSize().y);
		GuiSettings::setSettingsValue(Const::CpuMaxThreadsKey, getMaxThreads());
        GuiSettings::setSettingsValue(Const::GpuUniverseSizeXKey, getUniverseSizeForModelGpu().x);
        GuiSettings::setSettingsValue(Const::GpuUniverseSizeYKey, getUniverseSizeForModelGpu().y);
        GuiSettings::setSettingsValue(Const::GpuNumBlocksKey, getNumBlocks());
        GuiSettings::setSettingsValue(Const::GpuNumThreadsPerBlockKey, getNumThreadsPerBlock());
        GuiSettings::setSettingsValue(Const::GpuMaxClustersKey, getMaxClusters());
        GuiSettings::setSettingsValue(Const::GpuMaxCellsKey, getMaxCells());
        GuiSettings::setSettingsValue(Const::GpuMaxTokensKey, getMaxTokens());
        GuiSettings::setSettingsValue(Const::GpuMaxParticlesKey, getMaxParticles());
        GuiSettings::setSettingsValue(Const::GpuDynamicMemorySizeKey, getDynamicMemorySize());
        GuiSettings::setSettingsValue(Const::InitialEnergyKey, getEnergy());
        GuiSettings::setSettingsValue(Const::ModelComputationTypeKey, static_cast<int>(getModelType()));
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



