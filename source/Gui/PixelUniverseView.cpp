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
    _imageSectionItem = new ImageSectionItem(_viewport, QRectF(0,0, size.x, size.y));

    addItem(_imageSectionItem);

    QGraphicsScene::setSceneRect(0, 0, size.x, size.y);

    update();
}

void PixelUniverseView::activate()
{
	deactivate();
	_connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestData));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &PixelUniverseView::receivedNotifications));
	_connections.push_back(connect(_repository, &DataRepository::imageReady, this, &PixelUniverseView::displayData, Qt::QueuedConnection));
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
	requestData();
}

void PixelUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
{
	if (e->buttons() == Qt::MouseButton::LeftButton) {
		QVector2D pos(e->scenePos().x(), e->scenePos().y());
        QVector2D lastPos(e->lastScenePos().x(), e->lastScenePos().y());
        auto const force = (pos - lastPos) / 10;
        auto const action = boost::make_shared<_ApplyForceAction>(lastPos, pos, force);
        _access->applyAction(action);
	}
	if (e->buttons() == (Qt::MouseButton::LeftButton | Qt::MouseButton::RightButton)) {
        QVector2D pos(e->scenePos().x(), e->scenePos().y());
        QVector2D lastPos(e->lastScenePos().x(), e->lastScenePos().y());
        auto const force = (pos - lastPos) / 10;
        auto const action = boost::make_shared<_ApplyForceAction>(lastPos, pos, force);
        _access->applyAction(action);
	}
}

void PixelUniverseView::receivedNotifications(set<Receiver> const & targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}

	requestData();
}

void PixelUniverseView::requestData()
{
	IntRect rect = _viewport->getRect();
	_repository->requireImageFromSimulation(rect, _imageSectionItem->getImageOfVisibleRect());
}

void PixelUniverseView::displayData()
{
	update();
}

void PixelUniverseView::scrolled()
{
	requestData();
}

