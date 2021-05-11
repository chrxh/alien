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

    ui->horizontalScrollBar->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->verticalScrollBar->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->simulationView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    ui->horizontalScrollBar->setValue(0);
    ui->horizontalScrollBar->setRange(0, 0);
    ui->horizontalScrollBar->setPageStep(0);
    ui->verticalScrollBar->setValue(0);
    ui->verticalScrollBar->setRange(0, 0);
    ui->verticalScrollBar->setPageStep(0);
    ui->simulationView->setRenderHint(QPainter::Antialiasing, false);
    ui->simulationView->setRenderHint(QPainter::TextAntialiasing, false);
    ui->simulationView->setRenderHint(QPainter::SmoothPixmapTransform, false);
    ui->simulationView->setRenderHint(QPainter::LosslessImageRendering, false);
    ui->simulationView->setOptimizationFlags(
        QGraphicsView::DontSavePainterState | QGraphicsView::DontAdjustForAntialiasing | QGraphicsView::IndirectPainting);
    connect(ui->horizontalScrollBar, &QScrollBar::valueChanged, this, &SimulationViewWidget::horizontalScrolled);
    connect(ui->verticalScrollBar, &QScrollBar::valueChanged, this, &SimulationViewWidget::verticalScrolled);

    /*
    ui->simulationView->setViewport(new QOpenGLWidget());
    ui->simulationView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
*/
}

SimulationViewWidget::~SimulationViewWidget()
{
    delete ui;
}

void SimulationViewWidget::updateScrollbars(IntVector2D const& worldSize, QVector2D const& center, double zoom)
{
    _zoom = zoom;

    RealVector2D sceneSize = {toFloat(worldSize.x * zoom), toFloat(worldSize.y * zoom)};
    IntVector2D viewSize = {ui->simulationView->width(), ui->simulationView->height()};
    IntVector2D viewPos = {
        toInt(center.x() * zoom) - viewSize.x / 2, toInt(center.y() * zoom) - viewSize.y / 2};

    ui->horizontalScrollBar->blockSignals(true);
    ui->horizontalScrollBar->setValue(viewPos.x);
    ui->horizontalScrollBar->setRange(0, sceneSize.x - viewSize.x);
    ui->horizontalScrollBar->setPageStep(viewSize.x);
    ui->horizontalScrollBar->blockSignals(false);

    ui->verticalScrollBar->blockSignals(true);
    ui->verticalScrollBar->setValue(viewPos.y);
    ui->verticalScrollBar->setRange(0, sceneSize.y - viewSize.y);
    ui->verticalScrollBar->setPageStep(viewSize.y);
    ui->verticalScrollBar->blockSignals(false);
}

QGraphicsView* SimulationViewWidget::getGraphicsView() const
{
    return ui->simulationView;
}

void SimulationViewWidget::horizontalScrolled()
{
    Q_EMIT scrolledX(toFloat(ui->horizontalScrollBar->value() + ui->simulationView->width() / 2) / *_zoom);
}

void SimulationViewWidget::verticalScrolled()
{
    Q_EMIT scrolledY(toFloat(ui->verticalScrollBar->value() + ui->simulationView->height() / 2) / *_zoom);
}




