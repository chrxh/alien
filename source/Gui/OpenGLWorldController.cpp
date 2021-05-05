#include "OpenGLWorldController.h"

#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>
#include <QScrollBar>
#include <QResizeEvent>
#include <QtGui>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <QOpenGLShaderProgram>

#include <QtCore/qmath.h>

#include "Base/ServiceLocator.h"
#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/PhysicalActions.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SpaceProperties.h"

#include "Notifier.h"
#include "Settings.h"
#include "OpenGLScene.h"
#include "SimulationViewWidget.h"

OpenGLWorldController::OpenGLWorldController(SimulationViewWidget* simulationViewWidget, QObject* parent)
    : AbstractWorldController(simulationViewWidget, parent)
{
    _viewport = new QOpenGLWidget();

    auto graphicsView = _simulationViewWidget->getGraphicsView();
    graphicsView->setViewport(_viewport);
    graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    /*
    viewport->makeCurrent();
    _scene = new OpenGLUniverseScene(viewport->context(), this);
*/

    connect(&_updateViewTimer, &QTimer::timeout, this, &OpenGLWorldController::updateViewTimeout);
}

void OpenGLWorldController::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    disconnectView();
    _controller = controller;
    _repository = repository;
    _notifier = notifier;

    SET_CHILD(_access, access);

    auto graphicsView = _simulationViewWidget->getGraphicsView();
    auto width = graphicsView->width();
    auto height = graphicsView->height();

    if (!_scene) {
        _viewport->makeCurrent();
        _scene = new OpenGLScene(_viewport->context(), this);
    }
    _scene->init(access, repository->getImageMutex());
    _scene->resize({width, height});
    _scene->update();
    _scene->installEventFilter(this);
    graphicsView->installEventFilter(this);
}

void OpenGLWorldController::connectView()
{
    disconnectView();
    _connections.push_back(
        connect(_controller, &SimulationController::nextFrameCalculated, this, &OpenGLWorldController::requestImage));
    _connections.push_back(
        connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &OpenGLWorldController::receivedNotifications));
    _connections.push_back(connect(
        _repository, &DataRepository::imageReady, this, &OpenGLWorldController::imageReady, Qt::QueuedConnection));

    auto graphicsView = _simulationViewWidget->getGraphicsView();
    _connections.push_back(connect(
        graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &OpenGLWorldController::scrolled));
    _connections.push_back(
        connect(graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &OpenGLWorldController::scrolled));
}

void OpenGLWorldController::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void OpenGLWorldController::refresh()
{
    requestImage();
}

bool OpenGLWorldController::isActivated() const
{
    return _simulationViewWidget->getGraphicsView()->scene() == _scene;
}

void OpenGLWorldController::activate(double zoomFactor)
{
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    graphicsView->setScene(_scene);
    graphicsView->resetTransform();

    setZoomFactor(zoomFactor);
}

double OpenGLWorldController::getZoomFactor() const
{
    return _zoomFactor;
}

void OpenGLWorldController::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
    updateScrollbars();
}

void OpenGLWorldController::setZoomFactor(double zoomFactor, IntVector2D const& viewPos)
{
    auto worldPos = mapViewToWorldPosition(viewPos.toQVector2D());
    setZoomFactor(zoomFactor);
    centerTo(worldPos, viewPos);
}

QVector2D OpenGLWorldController::getCenterPositionOfScreen() const
{
    return _center;
}

void OpenGLWorldController::centerTo(QVector2D const& position)
{
    _center = position;
    updateScrollbars();
}

void OpenGLWorldController::centerTo(QVector2D const& worldPosition, IntVector2D const& viewPos)
{
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    QVector2D deltaViewPos{
        static_cast<float>(viewPos.x) - static_cast<float>(graphicsView->width()) / 2.0f,
        static_cast<float>(viewPos.y) - static_cast<float>(graphicsView->height()) / 2.0f};
    auto deltaWorldPos = mapDeltaViewToDeltaWorldPosition(deltaViewPos);
    centerTo(worldPosition - deltaWorldPos);
}

void OpenGLWorldController::updateScrollbars()
{
    _simulationViewWidget->updateScrollbars(
        _controller->getContext()->getSpaceProperties()->getSize(), getCenterPositionOfScreen(), getZoomFactor());
}

bool OpenGLWorldController::eventFilter(QObject* object, QEvent* event)
{
    if (object == _scene) {
        if (event->type() == QEvent::GraphicsSceneMousePress) {
            mousePressEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }

        if (event->type() == QEvent::GraphicsSceneMouseMove) {
            mouseMoveEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }

        if (event->type() == QEvent::GraphicsSceneMouseRelease) {
            mouseReleaseEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
        }
    }

    if (object = _simulationViewWidget->getGraphicsView()) {
        if (event->type() == QEvent::Resize) {
            resize(static_cast<QResizeEvent*>(event));
        }
    }
    return false;
}

void OpenGLWorldController::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    auto viewPos = IntVector2D{static_cast<int>(event->scenePos().x()), static_cast<int>(event->scenePos().y())};
    auto worldPos = mapViewToWorldPosition(viewPos.toQVector2D());

    if (event->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(viewPos);
    }
    if (event->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(viewPos);
    }
    if (event->buttons() == Qt::MouseButton::MiddleButton) {
        _worldPosForMovement = worldPos;
    }

    /*
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x() / _zoomFactor, event->scenePos().y() / _zoomFactor);
        _access->selectEntities(pos);
        requestImage();
    }
*/
}

void OpenGLWorldController::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    IntVector2D viewPos{static_cast<int>(event->scenePos().x()), static_cast<int>(event->scenePos().y())};
    if (event->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(viewPos);
    }
    if (event->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(viewPos);
    }
    if (event->buttons() == Qt::MouseButton::MiddleButton) {
        centerTo(*_worldPosForMovement, viewPos);
        refresh();
    }

    /*
    auto const pos = QVector2D(e->scenePos().x() / _zoomFactor, e->scenePos().y() / _zoomFactor);
    auto const lastPos = QVector2D(e->lastScenePos().x() / _zoomFactor, e->lastScenePos().y() / _zoomFactor);

    if (_controller->getRun()) {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyForceAction>(lastPos, pos, force));
        }
        if (e->buttons() == Qt::MouseButton::RightButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyRotationAction>(lastPos, pos, force));
        }
    }
    else {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const displacement = pos - lastPos;
            _access->applyAction(boost::make_shared<_MoveSelectionAction>(displacement));
            requestImage();
        }
    }
*/
}

void OpenGLWorldController::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _worldPosForMovement = boost::none;

    Q_EMIT endContinuousZoom();

/*
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
*/
}

void OpenGLWorldController::resize(QResizeEvent* event)
{
    auto size = event->size();
    _scene->resize({size.width(), size.height()});
    updateScrollbars();
}

void OpenGLWorldController::receivedNotifications(set<Receiver> const& targets)
{
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
        return;
    }

    requestImage();
}

void OpenGLWorldController::requestImage()
{
    if (!_connections.empty()) {
        auto graphicsView = _simulationViewWidget->getGraphicsView();
        auto topLeft = mapViewToWorldPosition(QVector2D(0, 0));
        auto bottomRight = mapViewToWorldPosition(QVector2D(graphicsView->width() - 1, graphicsView->height() - 1));
        RealRect worldRect{RealVector2D(topLeft), RealVector2D(bottomRight)};
        auto sceneRect = _scene->sceneRect();
        _repository->requireVectorImageFromSimulation(
            worldRect,
            _zoomFactor,
            _scene->getImageResource(),
            {static_cast<int>(sceneRect.width() + 0.5), static_cast<int>(sceneRect.height() + 0.5)});
    }
}

void OpenGLWorldController::imageReady()
{
    _scene->update();
    _updateViewTimer.start(Const::OpenGLViewUpdateInterval);
    _scheduledViewUpdates = Const::ViewUpdates;
}

void OpenGLWorldController::scrolled()
{
    requestImage();
}

void OpenGLWorldController::updateViewTimeout()
{
    if (_scheduledViewUpdates > 0) {
        _scene->update();
        --_scheduledViewUpdates;
    }
    if (_scheduledViewUpdates == 0) {
        _updateViewTimer.stop();
    }
}

QVector2D OpenGLWorldController::mapViewToWorldPosition(QVector2D const& viewPos) const
{
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    QVector2D relCenter(
        toFloat(graphicsView->width() / (2.0 * _zoomFactor)), toFloat(graphicsView->height() / (2.0 * _zoomFactor)));
    QVector2D relWorldPos(viewPos.x() / _zoomFactor, viewPos.y() / _zoomFactor);
    return _center - relCenter + relWorldPos;
}

QVector2D OpenGLWorldController::mapDeltaViewToDeltaWorldPosition(QVector2D const& viewPos) const
{
    return viewPos / _zoomFactor;
}
