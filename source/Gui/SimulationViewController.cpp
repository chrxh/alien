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
#include "ItemUniverseView.h"
#include "OpenGLUniverseView.h"
#include "QApplicationHelper.h"
#include "StartupController.h"
#include "SimulationViewWidget.h"


SimulationViewController::SimulationViewController(QWidget* parent)
    : QObject(parent)
{
    _simulationViewWidget = new SimulationViewWidget(parent);

    _openGLUniverse = new OpenGLUniverseView(_simulationViewWidget, this);
    _itemUniverse = new ItemUniverseView(_simulationViewWidget, this);
    connect(_openGLUniverse, &OpenGLUniverseView::startContinuousZoomIn, this, &SimulationViewController::continuousZoomIn);
    connect(
        _openGLUniverse, &OpenGLUniverseView::startContinuousZoomOut, this, &SimulationViewController::continuousZoomOut);
    connect(_openGLUniverse, &OpenGLUniverseView::endContinuousZoom, this, &SimulationViewController::endContinuousZoom);
    connect(_itemUniverse, &ItemUniverseView::startContinuousZoomIn, this, &SimulationViewController::continuousZoomIn);
    connect(_itemUniverse, &ItemUniverseView::startContinuousZoomOut, this, &SimulationViewController::continuousZoomOut);
    connect(_itemUniverse, &ItemUniverseView::endContinuousZoom, this, &SimulationViewController::endContinuousZoom);

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

    _openGLUniverse->init(notifier, controller, access, repository);
    _itemUniverse->init(notifier, controller, repository);

    _openGLUniverse->activate(InitialZoomFactor);

    auto size = _controller->getContext()->getSpaceProperties()->getSize();
    _openGLUniverse->centerTo({static_cast<float>(size.x) / 2, static_cast<float>(size.y) / 2});

    _openGLUniverse->connectView();
    _openGLUniverse->refresh();

    Q_EMIT zoomFactorChanged(InitialZoomFactor);
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
    getActiveUniverseView()->refresh();
}

ActiveView SimulationViewController::getActiveView() const
{
    if (_openGLUniverse->isActivated()) {
        return ActiveView::OpenGLScene;
    }
    if (_itemUniverse->isActivated()) {
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
    auto itemUniverseView = dynamic_cast<ItemUniverseView*>(activeUniverseView);
    CHECK(itemUniverseView);

    itemUniverseView->toggleCenterSelection(value);
}

UniverseView* SimulationViewController::getActiveUniverseView() const
{
    if (_openGLUniverse->isActivated()) {
        return _openGLUniverse;
    }
    if (_itemUniverse->isActivated()) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}

UniverseView* SimulationViewController::getView(ActiveView activeView) const
{
    if (ActiveView::OpenGLScene == activeView) {
        return _openGLUniverse;
    }
    if (ActiveView::ItemScene == activeView) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}
