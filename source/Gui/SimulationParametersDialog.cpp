#include <iostream>
#include <fstream>
#include <QFileDialog>
#include <QMessageBox>

#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/Serializer.h"
#include "ModelBasic/SerializationHelper.h"

#include "Settings.h"
#include "SimulationParametersDialog.h"
#include "SimulationConfig.h"
#include "ui_simulationparametersdialog.h"

SimulationParametersDialog::SimulationParametersDialog(SimulationConfig const& config, Serializer* serializer, QWidget *parent)
	: QDialog(parent), ui(new Ui::SimulationParametersDialog), _simulationParameters(config->parameters)
	, _serializer(serializer), _config(config)
{
    ui->setupUi(this);
    setFont(GuiSettings::getGlobalFont());
    ui->treeWidget->expandAll();
	ui->treeWidget->setColumnWidth(0, 270);

    updateWidgetsFromSimulationParameters ();

    //connections
    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(updateSimulationParametersFromWidgets()));
    connect(ui->defaultButton, SIGNAL(clicked()), this, SLOT(defaultButtonClicked()));
    connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
	connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &SimulationParametersDialog::okClicked);
}

SimulationParametersDialog::~SimulationParametersDialog()
{
    delete ui;
}

SimulationParameters const& SimulationParametersDialog::getSimulationParameters () const
{
    return _simulationParameters;
}

void SimulationParametersDialog::okClicked()
{
	string errorMsg;
	auto valResult = _config->validate(errorMsg);
	if (valResult == _SimulationConfig::ValidationResult::Ok) {
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

void SimulationParametersDialog::updateWidgetsFromSimulationParameters ()
{
	setItem("max radius", 0, _simulationParameters.clusterMaxRadius);

	setItem("mutation probability", 0, _simulationParameters.cellMutationProb);
	setItem("min distance", 0, _simulationParameters.cellMinDistance);
	setItem("max distance", 0, _simulationParameters.cellMaxDistance);
	setItem("mass", 0, 1.0/_simulationParameters.cellMass_Reciprocal);
	setItem("max force", 0, _simulationParameters.cellMaxForce);
    setItem("max force decay probability", 0, _simulationParameters.cellMaxForceDecayProb);
    setItem("max bonds", 0, _simulationParameters.cellMaxBonds);
    setItem("max token", 0, _simulationParameters.cellMaxToken);
    setItem("max token branch number", 0, _simulationParameters.cellMaxTokenBranchNumber);
    setItem("min energy", 0, _simulationParameters.cellMinEnergy);
    setItem("transformation probability", 0, _simulationParameters.cellTransformationProb);
    setItem("fusion velocity", 0, _simulationParameters.cellFusionVelocity);

    setItem("max instructions", 0, _simulationParameters.cellFunctionComputerMaxInstructions);
    setItem("memory size", 0, _simulationParameters.cellFunctionComputerCellMemorySize);
    setItem("offspring cell energy", 0, _simulationParameters.cellFunctionConstructorOffspringCellEnergy);
    setItem("offspring cell distance", 0, _simulationParameters.cellFunctionConstructorOffspringCellDistance);
	setItem("offspring token energy", 0, _simulationParameters.cellFunctionConstructorOffspringTokenEnergy);
    setItem("range", 0, _simulationParameters.cellFunctionSensorRange);
    setItem("strength", 0, _simulationParameters.cellFunctionWeaponStrength);
    setItem("range", 1, _simulationParameters.cellFunctionCommunicatorRange);

	setItem("memory size", 1, _simulationParameters.tokenMemorySize);
	setItem("min energy", 1, _simulationParameters.tokenMinEnergy);

    setItem("exponent", 0, _simulationParameters.radiationExponent);
    setItem("factor", 0, _simulationParameters.radiationFactor);
    setItem("probability", 0, _simulationParameters.radiationProb);
    setItem("velocity multiplier", 0, _simulationParameters.radiationVelocityMultiplier);
    setItem("velocity perturbation", 0, _simulationParameters.radiationVelocityPerturbation);
}

void SimulationParametersDialog::updateSimulationParametersFromWidgets ()
{
	_simulationParameters.clusterMaxRadius = getItemReal("max radius", 0);
	
	_simulationParameters.cellMutationProb = getItemReal("mutation probability", 0);
	_simulationParameters.cellMinDistance = getItemReal("min distance", 0);
	_simulationParameters.cellMaxDistance = getItemReal("max distance", 0);
    _simulationParameters.cellMass_Reciprocal = 1.0/ getItemReal("mass", 0);
    _simulationParameters.cellMaxForce = getItemReal("max force", 0);
    _simulationParameters.cellMaxForceDecayProb = getItemReal("max force decay probability", 0);
    _simulationParameters.cellMaxBonds = getItemInt("max bonds", 0);
    _simulationParameters.cellMaxToken = getItemInt("max token", 0);
    _simulationParameters.cellMaxTokenBranchNumber = getItemInt("max token branch number", 0);
    _simulationParameters.cellFunctionConstructorOffspringCellEnergy = getItemReal("offspring cell energy", 0);
    _simulationParameters.cellMinEnergy = getItemReal("min energy", 0);
    _simulationParameters.cellTransformationProb = getItemReal("transformation probability", 0);
    _simulationParameters.cellFusionVelocity = getItemReal("fusion velocity", 0);

    _simulationParameters.cellFunctionComputerMaxInstructions = getItemInt("max instructions", 0);
    _simulationParameters.cellFunctionComputerCellMemorySize = getItemInt("memory size", 0);
    _simulationParameters.cellFunctionConstructorOffspringCellDistance = getItemReal("offspring cell distance", 0);
    _simulationParameters.cellFunctionWeaponStrength = getItemReal("strength", 0);
    _simulationParameters.cellFunctionSensorRange = getItemReal("range", 0);
    _simulationParameters.cellFunctionCommunicatorRange = getItemReal("range", 1);

	_simulationParameters.tokenMemorySize = getItemInt("memory size", 1);
	_simulationParameters.cellFunctionConstructorOffspringTokenEnergy = getItemReal("offspring token energy", 0);
    _simulationParameters.tokenMinEnergy = getItemReal("min energy", 1);

    _simulationParameters.radiationExponent = getItemReal("exponent", 0);
    _simulationParameters.radiationFactor = getItemReal("factor", 0);
    _simulationParameters.radiationProb = getItemReal("probability", 0);
    _simulationParameters.radiationVelocityMultiplier = getItemReal("velocity multiplier", 0);
    _simulationParameters.radiationVelocityPerturbation = getItemReal("velocity perturbation", 0);
}

void SimulationParametersDialog::setItem(QString key, int matchPos, int value)
{
	ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->setText(1, QString("%1").arg(value));
}

void SimulationParametersDialog::setItem(QString key, int matchPos, qreal value)
{
	ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->setText(1, QString("%1").arg(value));
}

int SimulationParametersDialog::getItemInt(QString key, int matchPos)
{
	bool ok(true);
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(1).toInt(&ok);
}

qreal SimulationParametersDialog::getItemReal(QString key, int matchPos)
{
	bool ok(true);
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(1).toDouble(&ok);
}

bool SimulationParametersDialog::saveSimulationParameters(string filename)
{
	try {
		std::ofstream stream(filename, std::ios_base::out | std::ios_base::binary);
		string const& data = _serializer->serializeSimulationParameters(_simulationParameters);;
		size_t dataSize = data.size();
		stream.write(reinterpret_cast<char*>(&dataSize), sizeof(size_t));
		stream.write(&data[0], data.size());
		stream.close();
		if (stream.fail()) {
			return false;
		}
	}
	catch (...) {
		return false;
	}
	return true;
}

void SimulationParametersDialog::defaultButtonClicked ()
{
	_simulationParameters = ModelSettings::getDefaultSimulationParameters();
    updateWidgetsFromSimulationParameters();
}

void SimulationParametersDialog::loadButtonClicked ()
{
    QString filename = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !filename.isEmpty() ) {
		auto origSimulationParameters = _simulationParameters;
		if (SerializationHelper::loadFromFile<SimulationParameters>(filename.toStdString(), [&](string const& data) { return _serializer->deserializeSimulationParameters(data); }, _simulationParameters)) {
			updateWidgetsFromSimulationParameters();
		}
		else {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. The specified simulation parameter file could not loaded.");
			msgBox.exec();
			_simulationParameters = origSimulationParameters;
		}
    }
}

void SimulationParametersDialog::saveButtonClicked ()
{
    QString filename = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !filename.isEmpty() ) {
		updateSimulationParametersFromWidgets();
		if (!SerializationHelper::saveToFile(filename.toStdString(), [&]() { return _serializer->serializeSimulationParameters(_simulationParameters); })) {
			QMessageBox msgBox(QMessageBox::Critical, "Error", "An error occurred. Simulation parameters could not saved.");
			msgBox.exec();
		}
    }
}
