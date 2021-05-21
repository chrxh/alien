#include "SimulationViewController.h"

#include <QFile>
#include <QGraphicsView>
#include <QOpenGLWidget>
#include <QScrollBar>

#include <QTextStream>
#include <QTimer>

#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SpaceProperties.h"
#include "Gui/Settings.h"
#include "ItemWorldController.h"
#include "OpenGLWorldController.h"
#include "QApplicationHelper.h"
#include "StartupController.h"
#include "SimulationViewWidget.h"


SimulationViewController::SimulationViewController(QWidget* parent)
    : QObject(parent)
{
    _simulationViewWidget = new SimulationViewWidget(parent);

    _openGLWorld = new OpenGLWorldController(_simulationViewWidget, parent);
    _itemWorld = new ItemWorldController(_simulationViewWidget, this);
    connect(_openGLWorld, &OpenGLWorldController::startContinuousZoomIn, this, &SimulationViewController::continuousZoomIn);
    connect(
        _openGLWorld, &OpenGLWorldController::startContinuousZoomOut, this, &SimulationViewController::continuousZoomOut);
    connect(_openGLWorld, &OpenGLWorldController::endContinuousZoom, this, &SimulationViewController::endContinuousZoom);
    connect(_itemWorld, &ItemWorldController::startContinuousZoomIn, this, &SimulationViewController::continuousZoomIn);
    connect(_itemWorld, &ItemWorldController::startContinuousZoomOut, this, &SimulationViewController::continuousZoomOut);
    connect(_itemWorld, &ItemWorldController::endContinuousZoom, this, &SimulationViewController::endContinuousZoom);

    auto startupScene = new QGraphicsScene(this);
    startupScene->setBackgroundBrush(QBrush(Const::UniverseColor));
    _simulationViewWidget->getGraphicsView()->setScene(startupScene);
}

QWidget* SimulationViewController::getWidget() const
{
    return _simulationViewWidget;
}

void SimulationViewController::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    auto const InitialZoomFactor = 4.0;

    _controller = controller;

    _openGLWorld->init(notifier, controller, access, repository);
    _itemWorld->init(notifier, controller, repository);

    _openGLWorld->activate(InitialZoomFactor);

    auto size = _controller->getContext()->getSpaceProperties()->getSize();
    auto center = QVector2D{static_cast<float>(size.x) / 2, static_cast<float>(size.y) / 2};
    _openGLWorld->centerTo(center);

    _openGLWorld->connectView();
    _openGLWorld->refresh();
    _simulationViewWidget->updateScrollbars(size, center, InitialZoomFactor);

    Q_EMIT zoomFactorChanged(InitialZoomFactor);
}

void SimulationViewController::setSettings(SimulationViewSettings const& settings)
{
    _itemWorld->setSettings(settings);
    _openGLWorld->setSettings(settings);
}

void SimulationViewController::connectView()
{
    getActiveUniverseView()->connectView();
}

void SimulationViewController::disconnectView()
{
    getActiveUniverseView()->disconnectView();
}

void SimulationViewController::refresh()
{
    if (auto const& universeView = getActiveUniverseView()) {
        universeView->refresh();
    }
}

ActiveView SimulationViewController::getActiveView() const
{
    if (_openGLWorld->isActivated()) {
        return ActiveView::OpenGLScene;
    }
    if (_itemWorld->isActivated()) {
        return ActiveView::ItemScene;
    }

    THROW_NOT_IMPLEMENTED();
}

void SimulationViewController::setActiveScene(ActiveView activeScene)
{
    auto center = getActiveUniverseView()->getCenterPositionOfScreen();

    auto zoom = getZoomFactor();
    auto view = getView(activeScene);
    view->activate(zoom);

    getActiveUniverseView()->centerTo(center);
}

double SimulationViewController::getZoomFactor()
{
    return getActiveUniverseView()->getZoomFactor();
}

void SimulationViewController::setZoomFactor(double factor)
{
    auto activeView = getActiveUniverseView();
    auto screenCenterPos = activeView->getCenterPositionOfScreen();
    activeView->setZoomFactor(factor);
    activeView->centerTo(screenCenterPos);

    Q_EMIT zoomFactorChanged(factor);
}

void SimulationViewController::setZoomFactor(double zoomFactor, IntVector2D const& viewPos)
{
    getActiveUniverseView()->setZoomFactor(zoomFactor, viewPos);

    Q_EMIT zoomFactorChanged(zoomFactor);
}

QVector2D SimulationViewController::getViewCenterWithIncrement()
{
    auto screenCenterPos = getActiveUniverseView()->getCenterPositionOfScreen();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if (_posIncrement > 9.0) {
        _posIncrement = 0.0;
    }
    return screenCenterPos + posIncrement;
}

void SimulationViewController::toggleCenterSelection(bool value)
{
    auto activeUniverseView = getActiveUniverseView();
    auto itemUniverseView = dynamic_cast<ItemWorldController*>(activeUniverseView);
    CHECK(itemUniverseView);

    itemUniverseView->toggleCenterSelection(value);
}

AbstractWorldController* SimulationViewController::getActiveUniverseView() const
{
    if (_openGLWorld->isActivated()) {
        return _openGLWorld;
    }
    if (_itemWorld->isActivated()) {
        return _itemWorld;
    }

    return nullptr;
}

AbstractWorldController* SimulationViewController::getView(ActiveView activeView) const
{
    if (ActiveView::OpenGLScene == activeView) {
        return _openGLWorld;
    }
    if (ActiveView::ItemScene == activeView) {
        return _itemWorld;
    }

    THROW_NOT_IMPLEMENTED();
}
