#include "Monitor.h"
#include "ui_Monitor.h"

#include <QPaintEvent>

Monitor::Monitor(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Monitor)
{
    ui->setupUi(this);
}


Monitor::~Monitor()
{
    delete ui;
}

void Monitor::update (QMap< QString, qreal > data)
{
    ui->numberCellsLabel->setText(QString::number(data["cells"]));
    ui->numberClustersLabel->setText(QString::number(data["clusters"]));
    ui->numberEnergyParticlesLabel->setText(QString::number(data["energyParticles"]));
    ui->numberTokenLabel->setText(QString::number(data["token"]));
    ui->internalEnergyLabel->setText(QString::number(data["internalEnergy"],'f',2));
    ui->kinEnergyLabel->setText(QString::number(data["transEnergy"]+data["rotEnergy"],'f',2));
    ui->transEnergyLabel->setText(QString::number(data["transEnergy"],'f',2));
    ui->rotEnergyLabel->setText(QString::number(data["rotEnergy"],'f',2));
    ui->totalEnergyLabel->setText(QString::number(data["internalEnergy"]+data["transEnergy"]+data["rotEnergy"],'f',2));
}

bool Monitor::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}

