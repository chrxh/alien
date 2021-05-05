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

#include "OpenGLWorldController.h"
#include "ItemWorldController.h"
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

void SimulationViewWidget::updateScrollbars(IntVector2D const& worldSize, double zoom)
{
    RealVector2D sceneSize = {toFloat(worldSize.x * zoom), toFloat(worldSize.y * zoom)};
    IntVector2D viewSize = {ui->simulationView->width(), ui->simulationView->height()};
    ui->horizontalScrollBar->setRange(0, sceneSize.x - viewSize.x);
    ui->horizontalScrollBar->setPageStep(viewSize.x);
    ui->verticalScrollBar->setRange(0, sceneSize.y - viewSize.y);
    ui->verticalScrollBar->setPageStep(viewSize.y);
}

QGraphicsView* SimulationViewWidget::getGraphicsView() const
{
    return ui->simulationView;
}




