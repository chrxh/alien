#include <QFileDialog>
#include <QMessageBox>

#include "global/global.h"
#include "gui/guisettings.h"
#include "model/simulationparameters.h"

#include "simulationparametersdialog.h"
#include "ui_simulationparametersdialog.h"

SimulationParametersDialog::SimulationParametersDialog(SimulationParameters* parameters, QWidget *parent)
	: QDialog(parent), ui(new Ui::SimulationParametersDialog), _simulationParameters(*parameters)
{
    ui->setupUi(this);
    setFont(GuiFunctions::getGlobalFont());
    ui->treeWidget->expandAll();

    _localSimulationParameters = _simulationParameters;
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

void SimulationParametersDialog::updateSimulationParameters ()
{
    _simulationParameters = _localSimulationParameters;
}

void SimulationParametersDialog::setLocalSimulationParametersToWidgets ()
{
	setItem("mutation probability", 0, _localSimulationParameters.MUTATION_PROB);
	setItem("min distance", 0, _localSimulationParameters.CRIT_CELL_DIST_MIN);
	setItem("max distance", 0, _localSimulationParameters.CRIT_CELL_DIST_MAX);
	setItem("mass", 0, 1.0/_localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY);
	setItem("max force", 0, _localSimulationParameters.CELL_MAX_FORCE);
    setItem("max force decay probability", 0, _localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB);
    setItem("max bonds", 0, _localSimulationParameters.MAX_CELL_CONNECTIONS);
    setItem("max token", 0, _localSimulationParameters.CELL_TOKENSTACKSIZE);
    setItem("max token branch number", 0, _localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS);
    setItem("creation energy", 0, _localSimulationParameters.NEW_CELL_ENERGY);
    setItem("min energy", 0, _localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY);
    setItem("transformation probability", 0, _localSimulationParameters.CELL_TRANSFORM_PROB);
    setItem("fusion velocity", 0, _localSimulationParameters.CLUSTER_FUSION_VEL);

    setItem("max instructions", 0, _localSimulationParameters.CELL_NUM_INSTR);
    setItem("cell memory size", 0, _localSimulationParameters.CELL_MEMSIZE);
    setItem("token memory size", 0, _localSimulationParameters.TOKEN_MEMSIZE);
    setItem("offspring distance", 0, _localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST);
    setItem("strength", 0, _localSimulationParameters.CELL_WEAPON_STRENGTH);
    setItem("range", 0, _localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE);
    setItem("range", 1, _localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE);

    setItem("creation energy", 1, _localSimulationParameters.NEW_TOKEN_ENERGY);
	setItem("min energy", 1, _localSimulationParameters.MIN_TOKEN_ENERGY);

    setItem("exponent", 0, _localSimulationParameters.RAD_EXPONENT);
    setItem("factor", 0, _localSimulationParameters.RAD_FACTOR);
    setItem("probability", 0, _localSimulationParameters.RAD_PROBABILITY);
    setItem("velocity multiplier", 0, _localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT);
    setItem("velocity perturbation", 0, _localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB);
}

void SimulationParametersDialog::getLocalSimulationParametersFromWidgets ()
{
    _localSimulationParameters.MUTATION_PROB = getItemReal("mutation probability", 0);
	_localSimulationParameters.CRIT_CELL_DIST_MIN = getItemReal("min distance", 0);
	_localSimulationParameters.CRIT_CELL_DIST_MAX = getItemReal("max distance", 0);
    _localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY = 1.0/ getItemReal("mass", 0);
    _localSimulationParameters.CELL_MAX_FORCE = getItemReal("max force", 0);
    _localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB = getItemReal("max force decay probability", 0);
    _localSimulationParameters.MAX_CELL_CONNECTIONS = getItemInt("max bonds", 0);
    _localSimulationParameters.CELL_TOKENSTACKSIZE = getItemInt("max token", 0);
    _localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS = getItemInt("max token branch number", 0);
    _localSimulationParameters.NEW_CELL_ENERGY = getItemReal("creation energy", 0);
    _localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY = getItemReal("min energy", 0);
    _localSimulationParameters.CELL_TRANSFORM_PROB = getItemReal("transformation probability", 0);
    _localSimulationParameters.CLUSTER_FUSION_VEL = getItemReal("fusion velocity", 0);

    _localSimulationParameters.CELL_NUM_INSTR = getItemInt("max instructions", 0);
    _localSimulationParameters.CELL_MEMSIZE = getItemInt("cell memory size", 0);
    _localSimulationParameters.TOKEN_MEMSIZE = getItemInt("token memory size", 0);
    _localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST = getItemReal("offspring distance", 0);
    _localSimulationParameters.CELL_WEAPON_STRENGTH = getItemReal("strength", 0);
    _localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE = getItemReal("range", 0);
    _localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE = getItemReal("range", 1);

    _localSimulationParameters.NEW_TOKEN_ENERGY = getItemReal("creation energy", 1);
    _localSimulationParameters.MIN_TOKEN_ENERGY = getItemReal("min energy", 1);

    _localSimulationParameters.RAD_EXPONENT = getItemReal("exponent", 0);
    _localSimulationParameters.RAD_FACTOR = getItemReal("factor", 0);
    _localSimulationParameters.RAD_PROBABILITY = getItemReal("probability", 0);
    _localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT = getItemReal("velocity multiplier", 0);
    _localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB = getItemReal("velocity perturbation", 0);
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
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(matchPos).toInt(&ok);
}

qreal SimulationParametersDialog::getItemReal(QString key, int matchPos)
{
	bool ok(true);
	return ui->treeWidget->findItems(key, Qt::MatchExactly | Qt::MatchRecursive).at(matchPos)->text(matchPos).toDouble(&ok);
}

void SimulationParametersDialog::defaultButtonClicked ()
{
    _localSimulationParameters = SimulationParameters();
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
