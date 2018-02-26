#include "MonitorView.h"
#include "ui_MonitorView.h"

#include <QPaintEvent>

MonitorView::MonitorView(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MonitorView)
{
    ui->setupUi(this);
}


MonitorView::~MonitorView()
{
    delete ui;
}

void MonitorView::update (QMap< QString, qreal > data)
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

bool MonitorView::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}

