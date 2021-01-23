#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "ModelBasic/PhysicalActions.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "Gui/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/Notifier.h"

#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "ImageSectionItem.h"
#include "PixelUniverseView.h"

PixelUniverseView::PixelUniverseView(QObject* parent)
{
	setBackgroundBrush(QBrush(Const::BackgroundColor));
    update();
}

PixelUniverseView::~PixelUniverseView()
{
}

void PixelUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository,
    ViewportInterface* viewport)
{
	_controller = controller;
	_viewport = viewport;
	_repository = repository;
	_notifier = notifier;

    SET_CHILD(_access, access);

    delete _imageSectionItem;
    auto const viewportRect = _viewport->getRect();

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    _imageSectionItem = new ImageSectionItem(_viewport, QRectF(0,0, size.x*8, size.y*8), repository->getImageMutex());

    addItem(_imageSectionItem);

    QGraphicsScene::setSceneRect(0, 0, size.x*8, size.y*8);

    update();
}

void PixelUniverseView::activate()
{
	deactivate();
	_connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestImage));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &PixelUniverseView::receivedNotifications));
	_connections.push_back(connect(_repository, &DataRepository::imageReady, this, &PixelUniverseView::imageReady, Qt::QueuedConnection));
	_connections.push_back(connect(_viewport, &ViewportInterface::scrolled, this, &PixelUniverseView::scrolled));

	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    auto image = _imageSectionItem->getImageOfVisibleRect();
    _repository->requireImageFromSimulation(
        {{0, 0}, {image->width(), image->height()}}, _imageSectionItem->getImageOfVisibleRect());
}

void PixelUniverseView::deactivate()
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

void PixelUniverseView::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x(), event->scenePos().y());
        _access->selectEntities(pos);
        requestImage();
    }
}

void PixelUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
{
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
}

void PixelUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
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
	IntRect rect = _viewport->getRect();
	_repository->requireImageFromSimulation(rect, _imageSectionItem->getImageOfVisibleRect());
}

void PixelUniverseView::imageReady()
{
	update();
}

void PixelUniverseView::scrolled()
{
	requestImage();
}

