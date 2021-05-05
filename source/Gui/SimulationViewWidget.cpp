#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>
#include <QGraphicsBlurEffect>
#include <QFile>
#include <QTextStream>
#include <QOpenGLWidget>

#include "Gui/Settings.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SpaceProperties.h"

#include "OpenGLUniverseView.h"
#include "ItemUniverseView.h"
#include "QApplicationHelper.h"
#include "StartupController.h"

#include "SimulationViewWidget.h"
#include "ui_SimulationViewWidget.h"


SimulationViewWidget::SimulationViewWidget(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::SimulationViewWidget)
{
    ui->setupUi(this);

    ui->verticalScrollBar->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->horizontalScrollBar->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->simulationView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

/*
    ui->simulationView->setViewport(new QOpenGLWidget());
    ui->simulationView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
*/
}

SimulationViewWidget::~SimulationViewWidget()
{
    delete ui;
}

QGraphicsView* SimulationViewWidget::getGraphicsView() const
{
    return ui->simulationView;
}




