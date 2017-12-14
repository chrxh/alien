#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "gui/Settings.h"
#include "gui/visualeditor/ViewportInterface.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceProperties.h"

#include "DataController.h"
#include "PixelUniverseView.h"

PixelUniverseView::PixelUniverseView(QObject* parent)
{
	setBackgroundBrush(QBrush(BACKGROUND_COLOR));
    _pixmap = addPixmap(QPixmap());
    update();
}

PixelUniverseView::~PixelUniverseView()
{
	delete _image;
}

void PixelUniverseView::init(SimulationController* controller, DataController* manipulator, ViewportInterface* viewport)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_manipulator = manipulator;

	delete _image;
	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
	_image = new QImage(size.x, size.y, QImage::Format_RGB32);
	QGraphicsScene::setSceneRect(0, 0, _image->width(), _image->height());
}

void PixelUniverseView::activate()
{
	_connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestData));
	_connections.push_back(connect(_manipulator, &DataController::imageReady, this, &PixelUniverseView::retrieveAndDisplayData, Qt::QueuedConnection));
	_connections.push_back(connect(_viewport, &ViewportInterface::scrolling, this, &PixelUniverseView::scrolling));

	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
	_manipulator->requireImageFromSimulation({ { 0, 0 }, size }, _image);
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

void PixelUniverseView::requestData()
{
	IntRect rect = _viewport->getRect();
	_manipulator->requireImageFromSimulation(rect, _image);
}

void PixelUniverseView::retrieveAndDisplayData()
{
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

void PixelUniverseView::scrolling()
{
	requestData();
}

