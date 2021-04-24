#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QScrollBar>
#include <QGraphicsSceneMouseEvent>
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
#include "PixelImageSectionItem.h"
#include "PixelUniverseView.h"
#include "PixelViewport.h"

PixelUniverseView::PixelUniverseView(QGraphicsView* graphicsView, QObject* parent)
    : UniverseView(graphicsView, parent)
{
    _scene = new QGraphicsScene(parent);
    _scene->setBackgroundBrush(QBrush(Const::BackgroundColor));
    _scene->update();
    _scene->installEventFilter(this);
}

void PixelUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository)
{
    disconnectView();

    _controller = controller;

    delete _viewport;
    _viewport = new PixelViewport(_graphicsView, this);

	_repository = repository;
	_notifier = notifier;
    SET_CHILD(_access, access);

    delete _imageSectionItem;

    auto const size = _controller->getContext()->getSpaceProperties()->getSize();
    _imageSectionItem = new PixelImageSectionItem(_viewport, size, repository->getImageMutex());

    _scene->addItem(_imageSectionItem);
    _scene->setSceneRect(0, 0, size.x, size.y);
}

void PixelUniverseView::connectView()
{
    disconnectView();
    _connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestImage));
    _connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &PixelUniverseView::receivedNotifications));
    _connections.push_back(connect(_repository, &DataRepository::imageReady, this, &PixelUniverseView::imageReady, Qt::QueuedConnection));
    _connections.push_back(connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &PixelUniverseView::scrolled));
    _connections.push_back(connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &PixelUniverseView::scrolled));
}

void PixelUniverseView::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void PixelUniverseView::refresh()
{
	requestImage();
}

bool PixelUniverseView::isActivated() const
{
    return _graphicsView->scene() == _scene;
}

void PixelUniverseView::activate(double zoomFactor)
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _graphicsView->setScene(_scene);
    setZoomFactor(zoomFactor);
}

double PixelUniverseView::getZoomFactor() const
{
    return _zoomFactor;
}

void PixelUniverseView::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
    _graphicsView->resetTransform();
    _graphicsView->scale(_zoomFactor, _zoomFactor);
}

QVector2D PixelUniverseView::getCenterPositionOfScreen() const
{
    auto const result = _graphicsView->mapToScene(static_cast<double>(_graphicsView->width()) / 2.0, static_cast<double>(_graphicsView->height()) / 2.0);
    return{ static_cast<float>(result.x()), static_cast<float>(result.y()) };
}

void PixelUniverseView::centerTo(QVector2D const & position)
{
    centerToIntern(position);
}

bool PixelUniverseView::eventFilter(QObject* object, QEvent* event)
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

void PixelUniverseView::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
    auto pos = QVector2D(event->scenePos().x(), event->scenePos().y());
    if (event->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(pos);
    }
    if (event->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(pos);
    }
/*
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x(), event->scenePos().y());
        _access->selectEntities(pos);
        requestImage();
    }
*/
}

void PixelUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
{
/*
    auto const pos = QVector2D(e->scenePos().x(), e->scenePos().y());
    auto const lastPos = QVector2D(e->lastScenePos().x(), e->lastScenePos().y());

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

void PixelUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
    Q_EMIT endContinuousZoom();
/*
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
*/
}

void PixelUniverseView::receivedNotifications(set<Receiver> const & targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}

	requestImage();
}

void PixelUniverseView::requestImage()
{
    if (!_connections.empty()) {
        _repository->requirePixelImageFromSimulation(_viewport->getRect(), _imageSectionItem->getImageOfVisibleRect());
    }
}

void PixelUniverseView::imageReady()
{
	_scene->update();
}

void PixelUniverseView::scrolled()
{
	requestImage();
}

