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

    /*
    ui->simulationView->setViewport(new QOpenGLWidget());
    ui->simulationView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
*/

    _openGLUniverse = new OpenGLUniverseView(ui->simulationView, this);
    _itemUniverse = new ItemUniverseView(ui->simulationView, this);
    connect(_openGLUniverse, &OpenGLUniverseView::startContinuousZoomIn, this, &SimulationViewWidget::continuousZoomIn);
    connect(
        _openGLUniverse, &OpenGLUniverseView::startContinuousZoomOut, this, &SimulationViewWidget::continuousZoomOut);
    connect(_openGLUniverse, &OpenGLUniverseView::endContinuousZoom, this, &SimulationViewWidget::endContinuousZoom);
    connect(_itemUniverse, &ItemUniverseView::startContinuousZoomIn, this, &SimulationViewWidget::continuousZoomIn);
    connect(_itemUniverse, &ItemUniverseView::startContinuousZoomOut, this, &SimulationViewWidget::continuousZoomOut);
    connect(_itemUniverse, &ItemUniverseView::endContinuousZoom, this, &SimulationViewWidget::endContinuousZoom);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->verticalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);

    ui->simulationView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->simulationView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto startupScene = new QGraphicsScene(this);
    startupScene->setBackgroundBrush(QBrush(Const::UniverseColor));
    ui->simulationView->setScene(startupScene);
}

SimulationViewWidget::~SimulationViewWidget()
{
    delete ui;
}

void SimulationViewWidget::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    auto const InitialZoomFactor = 4.0;

	_controller = controller;

    _openGLUniverse->init(notifier, controller, access, repository);
    _itemUniverse->init(notifier, controller, repository);

    _openGLUniverse->activate(InitialZoomFactor);

    auto size = _controller->getContext()->getSpaceProperties()->getSize();
    _openGLUniverse->centerTo({ static_cast<float>(size.x) / 2, static_cast<float>(size.y) / 2 });

    _openGLUniverse->connectView();
    _openGLUniverse->refresh();

    Q_EMIT zoomFactorChanged(InitialZoomFactor);
}

void SimulationViewWidget::connectView()
{
    getActiveUniverseView()->connectView();
}

void SimulationViewWidget::disconnectView()
{
    getActiveUniverseView()->disconnectView();
}

void SimulationViewWidget::refresh()
{
    getActiveUniverseView()->refresh();
}

ActiveView SimulationViewWidget::getActiveView() const
{
    if (_openGLUniverse->isActivated()) {
        return ActiveView::OpenGLScene;
    }
    if (_itemUniverse->isActivated()) {
        return ActiveView::ItemScene;
    }

    THROW_NOT_IMPLEMENTED();
}

void SimulationViewWidget::setActiveScene (ActiveView activeScene)
{
    auto center = getActiveUniverseView()->getCenterPositionOfScreen();

    auto zoom = getZoomFactor();
    auto view = getView(activeScene);
    view->activate(zoom);

    getActiveUniverseView()->centerTo(center);
}

double SimulationViewWidget::getZoomFactor()
{
    return getActiveUniverseView()->getZoomFactor();
}

void SimulationViewWidget::setZoomFactor(double factor)
{
    auto activeView = getActiveUniverseView();
    auto screenCenterPos = activeView->getCenterPositionOfScreen();
    activeView->setZoomFactor(factor);
    activeView->centerTo(screenCenterPos);

    Q_EMIT zoomFactorChanged(factor);
}

void SimulationViewWidget::setZoomFactor(double zoomFactor, IntVector2D const& viewPos)
{
    getActiveUniverseView()->setZoomFactor(zoomFactor, viewPos);

    Q_EMIT zoomFactorChanged(zoomFactor);
}

QVector2D SimulationViewWidget::getViewCenterWithIncrement ()
{
	auto screenCenterPos = getActiveUniverseView()->getCenterPositionOfScreen();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if (_posIncrement > 9.0) {
        _posIncrement = 0.0;
    }
    return screenCenterPos + posIncrement;
}

void SimulationViewWidget::toggleCenterSelection(bool value)
{
    auto activeUniverseView = getActiveUniverseView();
    auto itemUniverseView = dynamic_cast<ItemUniverseView*>(activeUniverseView);
    CHECK(itemUniverseView);

    itemUniverseView->toggleCenterSelection(value);
}

UniverseView * SimulationViewWidget::getActiveUniverseView() const
{
    if (_openGLUniverse->isActivated()) {
        return _openGLUniverse;
    }
    if (_itemUniverse->isActivated()) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}

UniverseView * SimulationViewWidget::getView(ActiveView activeView) const
{
    if (ActiveView::OpenGLScene == activeView) {
        return _openGLUniverse;
    }
    if (ActiveView::ItemScene== activeView) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}



