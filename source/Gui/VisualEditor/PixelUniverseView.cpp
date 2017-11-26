#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "gui/Settings.h"
#include "gui/visualeditor/ViewportInterface.h"
#include "Model/Api/SimulationAccess.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceMetric.h"

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

void PixelUniverseView::init(SimulationController* controller, SimulationAccess* access, ViewportInterface* viewport)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_simAccess = access;

	delete _image;
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_image = new QImage(size.x, size.y, QImage::Format_RGB32);
	QGraphicsScene::setSceneRect(0, 0, _image->width(), _image->height());

}

void PixelUniverseView::activate()
{
	connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestData);
	connect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverseView::retrieveAndDisplayData, Qt::QueuedConnection);
	connect(_viewport, &ViewportInterface::scrolling, this, &PixelUniverseView::scrolling);

	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_simAccess->requireImage({ { 0, 0 }, size }, _image);
}

void PixelUniverseView::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverseView::requestData);
	disconnect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverseView::retrieveAndDisplayData);
	disconnect(_viewport, &ViewportInterface::scrolling, this, &PixelUniverseView::scrolling);
}

void PixelUniverseView::requestData()
{
	IntRect rect = _viewport->getRect();
	_simAccess->requireImage(rect, _image);
}

void PixelUniverseView::retrieveAndDisplayData()
{
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

void PixelUniverseView::scrolling()
{
	requestData();
}

