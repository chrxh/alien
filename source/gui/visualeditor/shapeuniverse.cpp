#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Gui/Settings.h"
#include "Model/SimulationController.h"
#include "Model/BuilderFacade.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "model/Context/SimulationContextApi.h"
#include "model/Context/SpaceMetric.h"

#include "ShapeUniverse.h"

ShapeUniverse::ShapeUniverse(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(UNIVERSE_COLOR));
}

ShapeUniverse::~ShapeUniverse()
{
}

void ShapeUniverse::init(SimulationController * controller, ViewportInfo * viewport)
{
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();
	_context = controller->getContext();
	_viewport = viewport;
	auto simAccess = facade->buildSimulationAccess(_context);
	SET_CHILD(_simAccess, simAccess);

	connect(_simAccess, &SimulationAccess::dataReadyToRetrieve, this, &ShapeUniverse::retrieveAndDisplayData);
}

void ShapeUniverse::setActive()
{
	IntVector2D size = _context->getSpaceMetric()->getSize();
	ResolveDescription resolveDesc;
	_simAccess->requireData({ { 0, 0 }, size }, resolveDesc);
}

void ShapeUniverse::retrieveAndDisplayData()
{
	auto const& dataDesc = _simAccess->retrieveData();

}


