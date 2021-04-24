#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QScrollBar>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "EngineInterface/PhysicalActions.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SpaceProperties.h"
#include "Gui/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/Notifier.h"

#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "VectorImageSectionItem.h"
#include "VectorUniverseView.h"
#include "VectorViewport.h"

VectorUniverseView::VectorUniverseView(QGraphicsView* graphicsView, QObject* parent)
    : UniverseView(graphicsView, parent)
{
    _scene = new QGraphicsScene(parent);
    _scene->setBackgroundBrush(QBrush(Const::BackgroundColor));
    _scene->update();
    _scene->installEventFilter(this);
}

void VectorUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    disconnectView();
    _controller = controller;
    _repository = repository;
    _notifier = notifier;

    delete _viewport;
    _viewport = new VectorViewport(_graphicsView, this);

    SET_CHILD(_access, access);

    delete _imageSectionItem;

    auto width = _graphicsView->width();
    auto height = _graphicsView->height();
    _imageSectionItem = new VectorImageSectionItem(_viewport, {width, height}, repository->getImageMutex());
    _scene->setSceneRect(0, 0, width - 1, height - 1);

    _scene->addItem(_imageSectionItem);
}

void VectorUniverseView::connectView()
{
    disconnectView();
    _connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &VectorUniverseView::requestImage));
    _connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &VectorUniverseView::receivedNotifications));
    _connections.push_back(connect(_repository, &DataRepository::imageReady, this, &VectorUniverseView::imageReady, Qt::QueuedConnection));
    _connections.push_back(connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &VectorUniverseView::scrolled));
    _connections.push_back(connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &VectorUniverseView::scrolled));
}

void VectorUniverseView::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void VectorUniverseView::refresh()
{
    requestImage();
}

bool VectorUniverseView::isActivated() const
{
    return _graphicsView->scene() == _scene;
}

void VectorUniverseView::activate(double zoomFactor)
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _graphicsView->setScene(_scene);
    _graphicsView->resetTransform();

    setZoomFactor(zoomFactor);
}

double VectorUniverseView::getZoomFactor() const
{
    return _zoomFactor;
}

void VectorUniverseView::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
    _viewport->setZoomFactor(zoomFactor);
    _imageSectionItem->setZoomFactor(zoomFactor);
/*
    auto const size = _controller->getContext()->getSpaceProperties()->getSize();
    _scene->setSceneRect(0, 0, static_cast<double>(size.x) * zoomFactor, static_cast<double>(size.y) * zoomFactor);
    _viewport->setZoomFactor(zoomFactor);
    _imageSectionItem->setZoomFactor(zoomFactor);
*/
}

void VectorUniverseView::setZoomFactor(double zoomFactor, QVector2D const& fixedPos)
{
    auto worldPosOfScreenCenter = getCenterPositionOfScreen();
    auto origZoomFactor = _zoomFactor;

    _zoomFactor = zoomFactor;
/*
    auto size = _controller->getContext()->getSpaceProperties()->getSize();
        _scene->setSceneRect(
            0, 0, static_cast<double>(size.x) * _zoomFactor, static_cast<double>(size.y) * _zoomFactor);
*/
    QVector2D mu(
        worldPosOfScreenCenter.x() * (zoomFactor / origZoomFactor - 1.0),
        worldPosOfScreenCenter.y() * (zoomFactor / origZoomFactor - 1.0));
    QVector2D correction(
        mu.x() * (worldPosOfScreenCenter.x() - fixedPos.x()) / worldPosOfScreenCenter.x(),
        mu.y() * (worldPosOfScreenCenter.y() - fixedPos.y()) / worldPosOfScreenCenter.y());
    centerTo(worldPosOfScreenCenter - correction);

    _viewport->setZoomFactor(_zoomFactor);
//    _imageSectionItem->setZoomFactor(_zoomFactor);
}

QVector2D VectorUniverseView::getCenterPositionOfScreen() const
{
    return _center;
/*
    auto width = static_cast<double>(_graphicsView->width());
    auto height = static_cast<double>(_graphicsView->height());
    auto scenePosition =
        _graphicsView->mapToScene(width / 2.0, height / 2.0);
    return{ static_cast<float>(scenePosition.x() / _zoomFactor), static_cast<float>(scenePosition.y() / _zoomFactor)};
*/
}

void VectorUniverseView::centerTo(QVector2D const & position)
{
    _center = position;
/*
    centerToIntern({static_cast<float>(position.x() * _zoomFactor), static_cast<float>(position.y() * _zoomFactor)});
*/
}

bool VectorUniverseView::eventFilter(QObject * object, QEvent * event)
{
    if (object != _scene) {
        return false;
    }

    if (event->type() == QEvent::GraphicsSceneMousePress) {
        mousePressEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    if (event->type() == QEvent::GraphicsSceneMouseMove) {
        mouseMoveEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    if (event->type() == QEvent::GraphicsSceneMouseRelease) {
        mouseReleaseEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    return false;
}

void VectorUniverseView::mousePressEvent(QGraphicsSceneMouseEvent * event)
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

void VectorUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
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

void VectorUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
    Q_EMIT endContinuousZoom();

/*
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
*/
}

void VectorUniverseView::receivedNotifications(set<Receiver> const & targets)
{
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
        return;
    }

    requestImage();
}

void VectorUniverseView::requestImage()
{
    if (!_connections.empty()) {
        auto topLeft = mapViewToWorldPosition(QVector2D(0, 0));
        auto bottomRight = mapViewToWorldPosition(QVector2D(_graphicsView->width() - 1, _graphicsView->height() - 1));
        RealRect rect{RealVector2D(topLeft), RealVector2D(bottomRight)};
/*
        RealVector2D upperLeft{
            static_cast<float>(std::max(0.0, _center.x() - _graphicsView->width() / (2.0 * _zoomFactor))),
            static_cast<float>(std::max(0.0, _center.y() - _graphicsView->height() / (2.0 * _zoomFactor)))};
        RealRect rect{
            {upperLeft.x, upperLeft.y},
            {upperLeft.x + static_cast<float>(_graphicsView->width() / _zoomFactor),
             upperLeft.y + static_cast<float>(_graphicsView->height() / _zoomFactor)}};
*/
        _repository->requireVectorImageFromSimulation(
            rect, _zoomFactor, _imageSectionItem->getImageOfVisibleRect());
    }
}

void VectorUniverseView::imageReady()
{
    _scene->update();
}

void VectorUniverseView::scrolled()
{
    requestImage();
}

QVector2D VectorUniverseView::mapViewToWorldPosition(QVector2D const& viewPos) const
{
    QVector2D relCenter(
        static_cast<float>(_graphicsView->width() / (2.0 * _zoomFactor)),
        static_cast<float>(_graphicsView->height() / (2.0 * _zoomFactor)));
    QVector2D relWorldPos(viewPos.x() / _zoomFactor, viewPos.y() / _zoomFactor);
    return _center - relCenter + relWorldPos;
}



