#include "simulationparametersdialog.h"
#include "ui_simulationparametersdialog.h"

#include <QFileDialog>
#include <QMessageBox>

#include "../../global/globalfunctions.h"
#include "../../global/simulationsettings.h"

SimulationParametersDialog::SimulationParametersDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SimulationParametersDialog)
{
    ui->setupUi(this);
    setFont(GlobalFunctions::getGlobalFont());
    ui->treeWidget->expandAll();

    localSimulationParameters = simulationParameters;
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
    simulationParameters = localSimulationParameters;
}

void SimulationParametersDialog::setLocalSimulationParametersToWidgets ()
{
    //load parameters
    ui->treeWidget->findItems("min distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CRIT_CELL_DIST_MIN));
    ui->treeWidget->findItems("max distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CRIT_CELL_DIST_MAX));
    ui->treeWidget->findItems("mass", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(1.0/localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY));
    ui->treeWidget->findItems("max force", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_MAX_FORCE));
    ui->treeWidget->findItems("max force decay probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB));
    ui->treeWidget->findItems("max bonds", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.MAX_CELL_CONNECTIONS));
    ui->treeWidget->findItems("max token", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_TOKENSTACKSIZE));
    ui->treeWidget->findItems("max token branch number", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS));
    ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.NEW_CELL_ENERGY));
    ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY));
    ui->treeWidget->findItems("transformation probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_TRANSFORM_PROB));
    ui->treeWidget->findItems("fusion velocity", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CLUSTER_FUSION_VEL));

    ui->treeWidget->findItems("max instructions", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_CODESIZE));
    ui->treeWidget->findItems("cell memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_MEMSIZE));
    ui->treeWidget->findItems("token memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.TOKEN_MEMSIZE));
    ui->treeWidget->findItems("offspring distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST));
    ui->treeWidget->findItems("strength", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_WEAPON_STRENGTH));
    ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE));
    ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE));

    ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(localSimulationParameters.NEW_TOKEN_ENERGY));
    ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->setText(1, QString("%1").arg(localSimulationParameters.MIN_TOKEN_ENERGY));

    ui->treeWidget->findItems("exponent", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.RAD_EXPONENT));
    ui->treeWidget->findItems("factor", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.RAD_FACTOR));
    ui->treeWidget->findItems("probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.RAD_PROBABILITY));
    ui->treeWidget->findItems("velocity multiplier", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT));
    ui->treeWidget->findItems("velocity perturbation", Qt::MatchExactly | Qt::MatchRecursive).at(0)->setText(1, QString("%1").arg(localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB));

}

void SimulationParametersDialog::getLocalSimulationParametersFromWidgets ()
{
    //read simulation parameters
    bool ok(true);
    localSimulationParameters.CRIT_CELL_DIST_MIN = ui->treeWidget->findItems("min distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CRIT_CELL_DIST_MAX = ui->treeWidget->findItems("max distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.INTERNAL_TO_KINETIC_ENERGY = 1.0/ui->treeWidget->findItems("mass", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_MAX_FORCE = ui->treeWidget->findItems("max force", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_MAX_FORCE_DECAY_PROB = ui->treeWidget->findItems("max force decay probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.MAX_CELL_CONNECTIONS = ui->treeWidget->findItems("max bonds", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.CELL_TOKENSTACKSIZE = ui->treeWidget->findItems("max token", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.MAX_TOKEN_ACCESS_NUMBERS = ui->treeWidget->findItems("max token branch number", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.NEW_CELL_ENERGY = ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CRIT_CELL_TRANSFORM_ENERGY = ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_TRANSFORM_PROB = ui->treeWidget->findItems("transformation probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CLUSTER_FUSION_VEL = ui->treeWidget->findItems("fusion velocity", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);

    localSimulationParameters.CELL_CODESIZE = ui->treeWidget->findItems("max instructions", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.CELL_MEMSIZE = ui->treeWidget->findItems("cell memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.TOKEN_MEMSIZE = ui->treeWidget->findItems("token memory size", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.CELL_FUNCTION_CONSTRUCTOR_OFFSPRING_DIST = ui->treeWidget->findItems("offspring distance", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toInt(&ok);
    localSimulationParameters.CELL_WEAPON_STRENGTH = ui->treeWidget->findItems("strength", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_FUNCTION_SENSOR_RANGE = ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE = ui->treeWidget->findItems("range", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);

    localSimulationParameters.NEW_TOKEN_ENERGY = ui->treeWidget->findItems("creation energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);
    localSimulationParameters.MIN_TOKEN_ENERGY = ui->treeWidget->findItems("min energy", Qt::MatchExactly | Qt::MatchRecursive).at(1)->text(1).toDouble(&ok);

    localSimulationParameters.RAD_EXPONENT = ui->treeWidget->findItems("exponent", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.RAD_FACTOR = ui->treeWidget->findItems("factor", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.RAD_PROBABILITY = ui->treeWidget->findItems("probability", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_RAD_ENERGY_VEL_MULT = ui->treeWidget->findItems("velocity multiplier", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
    localSimulationParameters.CELL_RAD_ENERGY_VEL_PERTURB = ui->treeWidget->findItems("velocity perturbation", Qt::MatchExactly | Qt::MatchRecursive).at(0)->text(1).toDouble(&ok);
}

void SimulationParametersDialog::defaultButtonClicked ()
{
    localSimulationParameters = SimulationParameters();
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
            localSimulationParameters.readData(in);
            file.close();

            //update widgets
            setLocalSimulationParametersToWidgets();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The specified simulation parameter file could not loaded.");
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
            localSimulationParameters.serializeData(out);
            file.close();
        }
        else {
            QMessageBox msgBox(QMessageBox::Warning,"Error", "An error occured. The simulation parameters could not saved.");
            msgBox.exec();
        }
    }
}
