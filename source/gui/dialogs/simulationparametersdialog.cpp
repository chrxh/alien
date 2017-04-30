#include <QFileDialog>
#include <QMessageBox>

#include "global/numbergenerator.h"
#include "gui/guisettings.h"
#include "model/context/simulationparameters.h"
#include "model/modelsettings.h"

#include "simulationparametersdialog.h"
#include "ui_simulationparametersdialog.h"

SimulationParametersDialog::SimulationParametersDialog(SimulationParameters* parameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::SimulationParametersDialog), _localSimulationParameters(*parameters)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());
    ui->treeWidget->expandAll();
	ui->treeWidget->setColumnWidth(0, 270);

    setLocalSimulationParametersToWidgets ();

    //connections
    connect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(getLocalSimulationParametersFromWidgets()));
    connect(ui->defaultButton, SIGNAL(clicked()), this, SLOT(defaultButtonClicked()));
    connect(ui->loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
    connect(ui->saveButton, SIGNAL(clicked()), this, SLOT(saveButtonClicked()));
}

SimulationParametersDialog::~SimulationParametersDialog()
{
    delete ui;
}

SimulationParameters SimulationParametersDialog::getSimulationParameters ()
{
    return _localSimulationParameters;
}

void SimulationParametersDialog::setLocalSimulationParametersToWidgets ()
{
	setItem("mutation probability", 0, _localSimulationParameters.cellMutationProb);
	setItem("min distance", 0, _localSimulationParameters.cellMinDistance);
	setItem("max distance", 0, _localSimulationParameters.cellMaxDistance);
	setItem("mass", 0, 1.0/_localSimulationParameters.cellMass_Reciprocal);
	setItem("max force", 0, _localSimulationParameters.callMaxForce);
    setItem("max force decay probability", 0, _localSimulationParameters.cellMaxForceDecayProb);
    setItem("max bonds", 0, _localSimulationParameters.cellMaxBonds);
    setItem("max token", 0, _localSimulationParameters.cellMaxToken);
    setItem("max token branch number", 0, _localSimulationParameters.cellMaxTokenBranchNumber);
    setItem("creation energy", 0, _localSimulationParameters.cellCreationEnergy);
    setItem("min energy", 0, _localSimulationParameters.cellMinEnergy);
    setItem("transformation probability", 0, _localSimulationParameters.cellTransformationProb);
    setItem("fusion velocity", 0, _localSimulationParameters.cellFusionVelocity);

    setItem("max instructions", 0, _localSimulationParameters.cellFunctionComputerMaxInstructions);
    setItem("memory size", 0, _localSimulationParameters.cellFunctionComputerCellMemorySize);
    setItem("offspring distance", 0, _localSimulationParameters.cellFunctionConstructorOffspringDistance);
    setItem("range", 0, _localSimulationParameters.cellFunctionSensorRange);
    setItem("strength", 0, _localSimulationParameters.cellFunctionWeaponStrength);
    setItem("range", 1, _localSimulationParameters.cellFunctionCommunicatorRange);

	setItem("memory size", 1, _localSimulationParameters.tokenMemorySize);
	setItem("creation energy", 1, _localSimulationParameters.tokenCreationEnergy);
	setItem("min energy", 1, _localSimulationParameters.tokenMinEnergy);

    setItem("exponent", 0, _localSimulationParameters.radiationExponent);
    setItem("factor", 0, _localSimulationParameters.radiationFactor);
    setItem("probability", 0, _localSimulationParameters.radiationProb);
    setItem("velocity multiplier", 0, _localSimulationParameters.radiationVelocityMultiplier);
    setItem("velocity perturbation", 0, _localSimulationParameters.radiationVelocityPerturbation);
}

void SimulationParametersDialog::getLocalSimulationParametersFromWidgets ()
{
    _localSimulationParameters.cellMutationProb = getItemReal("mutation probability", 0);
	_localSimulationParameters.cellMinDistance = getItemReal("min distance", 0);
	_localSimulationParameters.cellMaxDistance = getItemReal("max distance", 0);
    _localSimulationParameters.cellMass_Reciprocal = 1.0/ getItemReal("mass", 0);
    _localSimulationParameters.callMaxForce = getItemReal("max force", 0);
    _localSimulationParameters.cellMaxForceDecayProb = getItemReal("max force decay probability", 0);
    _localSimulationParameters.cellMaxBonds = getItemInt("max bonds", 0);
    _localSimulationParameters.cellMaxToken = getItemInt("max token", 0);
    _localSimulationParameters.cellMaxTokenBranchNumber = getItemInt("max token branch number", 0);
    _localSimulationParameters.cellCreationEnergy = getItemReal("creation energy", 0);
    _localSimulationParameters.cellMinEnergy = getItemReal("min energy", 0);
    _localSimulationParameters.cellTransformationProb = getItemReal("transformation probability", 0);
    _localSimulationParameters.cellFusionVelocity = getItemReal("fusion velocity", 0);

    _localSimulationParameters.cellFunctionComputerMaxInstructions = getItemInt("max instructions", 0);
    _localSimulationParameters.cellFunctionComputerCellMemorySize = getItemInt("memory size", 0);
    _localSimulationParameters.cellFunctionConstructorOffspringDistance = getItemReal("offspring distance", 0);
    _localSimulationParameters.cellFunctionWeaponStrength = getItemReal("strength", 0);
    _localSimulationParameters.cellFunctionSensorRange = getItemReal("range", 0);
    _localSimulationParameters.cellFunctionCommunicatorRange = getItemReal("range", 1);

	_localSimulationParameters.tokenMemorySize = getItemInt("memory size", 1);
	_localSimulationParameters.tokenCreationEnergy = getItemReal("creation energy", 1);
    _localSimulationParameters.tokenMinEnergy = getItemReal("min energy", 1);

    _localSimulationParameters.radiationExponent = getItemReal("exponent", 0);
    _localSimulationParameters.radiationFactor = getItemReal("factor", 0);
    _localSimulationParameters.radiationProb = getItemReal("probability", 0);
    _localSimulationParameters.radiationVelocityMultiplier = getItemReal("velocity multiplier", 0);
    _localSimulationParameters.radiationVelocityPerturbation = getItemReal("velocity perturbation", 0);
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

void SimulationParametersDialog::defaultButtonClicked ()
{
    ModelData::loadDefaultSimulationParameters(&_localSimulationParameters);
    setLocalSimulationParametersToWidgets();
}

void SimulationParametersDialog::loadButtonClicked ()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Load Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::ReadOnly) ) {

            //read simulation data
            QDataStream in(&file);
            _localSimulationParameters.deserializePrimitives(in);
            file.close();

            //update widgets
            setLocalSimulationParametersToWidgets();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The specified simulation parameter file could not loaded.");
            msgBox.exec();
        }
    }

}

void SimulationParametersDialog::saveButtonClicked ()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Save Simulation Parameters", "", "Alien Simulation Parameters(*.par)");
    if( !fileName.isEmpty() ) {
        QFile file(fileName);
        if( file.open(QIODevice::WriteOnly) ) {

            //serialize symbol table
            QDataStream out(&file);
            getLocalSimulationParametersFromWidgets();
            _localSimulationParameters.serializePrimitives(out);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occurred. The simulation parameters could not saved.");
            msgBox.exec();
        }
    }
}
