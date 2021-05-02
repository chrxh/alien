#include "OpenGLUniverseView.h"

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
#include "OpenGLUniverseScene.h"

OpenGLUniverseView::OpenGLUniverseView(QGraphicsView* graphicsView, QObject* parent)
    : UniverseView(graphicsView, parent)
{
    _viewport = new QOpenGLWidget();

    _graphicsView->setViewport(_viewport);
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
/*
    viewport->makeCurrent();
    _scene = new OpenGLUniverseScene(viewport->context(), this);
*/
}

void OpenGLUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    /*
    _graphicsView->setViewport(new QOpenGLWidget());
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
*/

    disconnectView();
    _controller = controller;
    _repository = repository;
    _notifier = notifier;

    SET_CHILD(_access, access);

    auto width = _graphicsView->width();
    auto height = _graphicsView->height();

    if (!_scene) {
        _viewport->makeCurrent();
        _scene = new OpenGLUniverseScene(_viewport->context(), this);
    }
    _scene->init(access, repository->getImageMutex());
    _scene->resize({width, height});
        /*
    delete _scene;
    _scene = new OpenGLUniverseScene(access, IntVector2D{width, height}, repository->getImageMutex(), this);
*/
    _scene->update();
    _scene->installEventFilter(this);
    _graphicsView->installEventFilter(this);
}

void OpenGLUniverseView::connectView()
{
    disconnectView();
    _connections.push_back(
        connect(_controller, &SimulationController::nextFrameCalculated, this, &OpenGLUniverseView::requestImage));
    _connections.push_back(
        connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &OpenGLUniverseView::receivedNotifications));
    _connections.push_back(
        connect(_repository, &DataRepository::imageReady, this, &OpenGLUniverseView::imageReady, Qt::QueuedConnection));
    _connections.push_back(
        connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &OpenGLUniverseView::scrolled));
    _connections.push_back(
        connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &OpenGLUniverseView::scrolled));
}

void OpenGLUniverseView::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void OpenGLUniverseView::refresh()
{
    requestImage();
}

bool OpenGLUniverseView::isActivated() const
{
    return _graphicsView->scene() == _scene;
}

void OpenGLUniverseView::activate(double zoomFactor)
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _graphicsView->setScene(_scene);
    _graphicsView->resetTransform();

    setZoomFactor(zoomFactor);
}

double OpenGLUniverseView::getZoomFactor() const
{
    return _zoomFactor;
}

void OpenGLUniverseView::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
}

void OpenGLUniverseView::setZoomFactor(double zoomFactor, QVector2D const& fixedPos)
{
    auto worldPosOfScreenCenter = getCenterPositionOfScreen();
    auto origZoomFactor = _zoomFactor;

    _zoomFactor = zoomFactor;
    QVector2D mu(
        worldPosOfScreenCenter.x() * (zoomFactor / origZoomFactor - 1.0),
        worldPosOfScreenCenter.y() * (zoomFactor / origZoomFactor - 1.0));
    QVector2D correction(
        mu.x() * (worldPosOfScreenCenter.x() - fixedPos.x()) / worldPosOfScreenCenter.x(),
        mu.y() * (worldPosOfScreenCenter.y() - fixedPos.y()) / worldPosOfScreenCenter.y());
    centerTo(worldPosOfScreenCenter - correction);
}

QVector2D OpenGLUniverseView::getCenterPositionOfScreen() const
{
    return _center;
}

void OpenGLUniverseView::centerTo(QVector2D const& position)
{
    _center = position;
}

bool OpenGLUniverseView::eventFilter(QObject* object, QEvent* event)
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

    if (object = _graphicsView) {
        if (event->type() == QEvent::Resize) {
            resize(static_cast<QResizeEvent*>(event));
        }
    }
    return false;
}

void OpenGLUniverseView::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    auto pos = mapViewToWorldPosition(QVector2D(event->scenePos().x(), event->scenePos().y()));

    if (event->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(pos);
    }
    if (event->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(pos);
    }
    /*
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x() / _zoomFactor, event->scenePos().y() / _zoomFactor);
        _access->selectEntities(pos);
        requestImage();
    }
*/
}

void OpenGLUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
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

void OpenGLUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    Q_EMIT endContinuousZoom();

/*
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
*/
}

void OpenGLUniverseView::resize(QResizeEvent* event)
{
    auto size = event->size();
    _scene->resize({size.width(), size.height()});
}

void OpenGLUniverseView::receivedNotifications(set<Receiver> const& targets)
{
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
        return;
    }

    requestImage();
}

void OpenGLUniverseView::requestImage()
{
    if (!_connections.empty()) {
        auto topLeft = mapViewToWorldPosition(QVector2D(0, 0));
        auto bottomRight = mapViewToWorldPosition(QVector2D(_graphicsView->width() - 1, _graphicsView->height() - 1));
        RealRect worldRect{RealVector2D(topLeft), RealVector2D(bottomRight)};
        auto sceneRect = _scene->sceneRect();
        _repository->requireVectorImageFromSimulation(
            worldRect,
            _zoomFactor,
            _scene->getImageResource(),
            {static_cast<int>(sceneRect.width() + 0.5), static_cast<int>(sceneRect.height() + 0.5)});
    }
}

void OpenGLUniverseView::imageReady()
{
    _scene->update();
}

void OpenGLUniverseView::scrolled()
{
    requestImage();
}

QVector2D OpenGLUniverseView::mapViewToWorldPosition(QVector2D const& viewPos) const
{
    QVector2D relCenter(
        static_cast<float>(_graphicsView->width() / (2.0 * _zoomFactor)),
        static_cast<float>(_graphicsView->height() / (2.0 * _zoomFactor)));
    QVector2D relWorldPos(viewPos.x() / _zoomFactor, viewPos.y() / _zoomFactor);
    return _center - relCenter + relWorldPos;
}
