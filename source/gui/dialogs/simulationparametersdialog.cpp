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
    //load parameters
    ui->treeWidget->findItems("min distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CRIT_CELL_DIST_MIN));
    ui->treeWidget->findItems("max distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CRIT_CELL_DIST_MAX));
    ui->treeWidget->findItems("mass", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(1.0/_localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY));
    ui->treeWidget->findItems("max force", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_MAX_FORCE));
    ui->treeWidget->findItems("max force decay probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB));
    ui->treeWidget->findItems("max bonds", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.MAX_CELL_CONNECTIONS));
    ui->treeWidget->findItems("max token", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_TOKENSTACKSIZE));
    ui->treeWidget->findItems("max token branch number", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS));
    ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.NEW_CELL_ENERGY));
    ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY));
    ui->treeWidget->findItems("transformation probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_TRANSFORM_PROB));
    ui->treeWidget->findItems("fusion velocity", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CLUSTER_FUSION_VEL));

    ui->treeWidget->findItems("max instructions", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_NUM_INSTR));
    ui->treeWidget->findItems("cell memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_MEMSIZE));
    ui->treeWidget->findItems("token memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.TOKEN_MEMSIZE));
    ui->treeWidget->findItems("offspring distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST));
    ui->treeWidget->findItems("strength", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_WEAPON_STRENGTH));
    ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE));
    ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE));

    ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(_localSimulationParameters.NEW_TOKEN_ENERGY));
    ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(_localSimulationParameters.MIN_TOKEN_ENERGY));

    ui->treeWidget->findItems("exponent", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.RAD_EXPONENT));
    ui->treeWidget->findItems("factor", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.RAD_FACTOR));
    ui->treeWidget->findItems("probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.RAD_PROBABILITY));
    ui->treeWidget->findItems("velocity multiplier", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT));
    ui->treeWidget->findItems("velocity perturbation", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(_localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB));

}

void SimulationParametersDialog::getLocalSimulationParametersFromWidgets ()
{
    //read simulation parameters
    bool ok(true);
    _localSimulationParameters.CRIT_CELL_DIST_MIN = ui->treeWidget->findItems("min distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CRIT_CELL_DIST_MAX = ui->treeWidget->findItems("max distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY = 1.0/ui->treeWidget->findItems("mass", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_MAX_FORCE = ui->treeWidget->findItems("max force", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB = ui->treeWidget->findItems("max force decay probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.MAX_CELL_CONNECTIONS = ui->treeWidget->findItems("max bonds", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.CELL_TOKENSTACKSIZE = ui->treeWidget->findItems("max token", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS = ui->treeWidget->findItems("max token branch number", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.NEW_CELL_ENERGY = ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY = ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_TRANSFORM_PROB = ui->treeWidget->findItems("transformation probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CLUSTER_FUSION_VEL = ui->treeWidget->findItems("fusion velocity", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);

    _localSimulationParameters.CELL_NUM_INSTR = ui->treeWidget->findItems("max instructions", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.CELL_MEMSIZE = ui->treeWidget->findItems("cell memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.TOKEN_MEMSIZE = ui->treeWidget->findItems("token memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST = ui->treeWidget->findItems("offspring distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    _localSimulationParameters.CELL_WEAPON_STRENGTH = ui->treeWidget->findItems("strength", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE = ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE = ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);

    _localSimulationParameters.NEW_TOKEN_ENERGY = ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);
    _localSimulationParameters.MIN_TOKEN_ENERGY = ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);

    _localSimulationParameters.RAD_EXPONENT = ui->treeWidget->findItems("exponent", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.RAD_FACTOR = ui->treeWidget->findItems("factor", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.RAD_PROBABILITY = ui->treeWidget->findItems("probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT = ui->treeWidget->findItems("velocity multiplier", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    _localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB = ui->treeWidget->findItems("velocity perturbation", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
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
