#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceProperties.h"
#include "Gui/VisualEditor/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/Notifier.h"

#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "Manipulator.h"
#include "PixelUniverseView.h"

PixelUniverseView::PixelUniverseView(QObject* parent)
{
	setBackgroundBrush(QBrush(Const::BackgroundColor));
    _pixmap = addPixmap(QPixmap());
	_manipulator = new Manipulator(this);
    update();
}

PixelUniverseView::~PixelUniverseView()
{
	delete _image;
}

void PixelUniverseView::init(Notifier* notifier, SimulationController* controller, DataRepository* repository
	, ViewportInterface* viewport)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_repository = repository;
	_notifier = notifier;

	_manipulator->init(controller->getContext());

	delete _image;
	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
	_image = new QImage(size.x, size.y, QImage::Format_RGB32);
	QGraphicsScene::setSceneRect(0, 0, _image->width(), _image->height());
}

void PixelUniverseView::activate()
{
	_connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestData));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &PixelUniverseView::receivedNotifications));
	_connections.push_back(connect(_repository, &DataRepository::imageReady, this, &PixelUniverseView::retrieveAndDisplayData, Qt::QueuedConnection));
	_connections.push_back(connect(_viewport, &ViewportInterface::scrolled, this, &PixelUniverseView::scrolled));

	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
	_repository->requireImageFromSimulation({ { 0, 0 }, size }, _image);
}

void PixelUniverseView::deactivate()
{
	for (auto const& connection : _connections) {
		disconnect(connection);
	}
}

void PixelUniverseView::refresh()
{
	requestData();
}

void PixelUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
{
	if (e->buttons() == Qt::MouseButton::LeftButton) {
		auto pos = e->scenePos();
		auto lastPos = e->lastScenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		_manipulator->applyForce({ static_cast<float>(pos.x()), static_cast<float>(pos.y()) }, delta);
	}
	if (e->buttons() == (Qt::MouseButton::LeftButton | Qt::MouseButton::RightButton)) {
		auto pos = e->scenePos();
		auto lastPos = e->lastScenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		_manipulator->applyRotation({ static_cast<float>(pos.x()), static_cast<float>(pos.y()) }, delta);
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
	_repository->requireImageFromSimulation(rect, _image);
}

void PixelUniverseView::retrieveAndDisplayData()
{
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

void PixelUniverseView::scrolled()
{
	requestData();
}

