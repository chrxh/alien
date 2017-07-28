#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Gui/Settings.h"
#include "Gui/visualeditor/ViewportInterface.h"
#include "Model/SimulationController.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Context/SimulationContextApi.h"
#include "Model/SpaceMetricApi.h"

#include "ShapeUniverse.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "VisualDescription.h"
#include "CoordinateSystem.h"

ShapeUniverse::ShapeUniverse(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(UNIVERSE_COLOR));
}

ShapeUniverse::~ShapeUniverse()
{
}

void ShapeUniverse::init(SimulationController * controller, SimulationAccess* access, ViewportInterface * viewport)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_simAccess = access;

	auto items = new ItemManager();
	auto descManager = new VisualDescription();
	SET_CHILD(_items, items);
	SET_CHILD(_visualDesc, descManager);

	items->init(this, viewport, _controller->getContext()->getSimulationParameters());
}

void ShapeUniverse::activate()
{
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_items->activate(size);

	connect(_controller, &SimulationController::nextFrameCalculated, this, &ShapeUniverse::requestData);
	connect(_simAccess, &SimulationAccess::dataReadyToRetrieve, this, &ShapeUniverse::retrieveAndDisplayData, Qt::QueuedConnection);

	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	_simAccess->requireData({ { 0, 0 }, size }, resolveDesc);
}

void ShapeUniverse::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &ShapeUniverse::requestData);
	disconnect(_simAccess, &SimulationAccess::dataReadyToRetrieve, this, &ShapeUniverse::retrieveAndDisplayData);
}

void ShapeUniverse::requestData()
{
	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	IntRect rect = _viewport->getRect();
	_simAccess->requireData(rect, resolveDesc);
}

void ShapeUniverse::retrieveAndDisplayData()
{
	_visualDesc->setData(_simAccess->retrieveData());
	_items->update(_visualDesc);
}

void ShapeUniverse::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
	auto items = QGraphicsScene::items(e->scenePos()).toStdList();
	set<uint64_t> cellIds;
	set<uint64_t> particleIds;
	for (auto item : items) {
		if (auto cellItem = qgraphicsitem_cast<CellItem*>(item)) {
			cellIds.insert(cellItem->getId());
		}
		if (auto particleItem = qgraphicsitem_cast<ParticleItem*>(item)) {
			particleIds.insert(particleItem->getId());
		}
	}
	_visualDesc->setSelection(cellIds, particleIds);
	_items->update(_visualDesc);
}

void ShapeUniverse::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
	bool leftButton = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
	bool rightButton = ((e->buttons() & Qt::RightButton) == Qt::RightButton);
	
	if (leftButton) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		_visualDesc->moveSelection(delta);
		_items->update(_visualDesc);
	}
}



