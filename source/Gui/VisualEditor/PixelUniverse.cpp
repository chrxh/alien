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

#include "PixelUniverse.h"

PixelUniverse::PixelUniverse(QObject* parent)
{
	setBackgroundBrush(QBrush(BACKGROUND_COLOR));
    _pixmap = addPixmap(QPixmap());
    update();
}

PixelUniverse::~PixelUniverse()
{
	delete _image;
}

void PixelUniverse::init(SimulationController* controller, SimulationAccess* access, ViewportInterface* viewport)
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

void PixelUniverse::activate()
{
	connect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverse::requestData);
	connect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverse::retrieveAndDisplayData, Qt::QueuedConnection);

	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_simAccess->requireImage({ { 0, 0 }, size }, _image);
}

void PixelUniverse::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &PixelUniverse::requestData);
	disconnect(_simAccess, &SimulationAccess::imageReady, this, &PixelUniverse::retrieveAndDisplayData);
}

void PixelUniverse::requestData()
{
	IntRect rect = _viewport->getRect();
	_simAccess->requireImage(rect, _image);
}

void PixelUniverse::retrieveAndDisplayData()
{
	_pixmap->setPixmap(QPixmap::fromImage(*_image));
}

