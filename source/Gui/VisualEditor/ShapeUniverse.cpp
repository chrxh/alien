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
	SET_CHILD(_itemManager, items);
	SET_CHILD(_visualDesc, descManager);

	items->init(this, viewport, _controller->getContext()->getSimulationParameters());
}

void ShapeUniverse::activate()
{
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_itemManager->activate(size);

	connect(_controller, &SimulationController::nextFrameCalculated, this, &ShapeUniverse::requestData);
	connect(_simAccess, &SimulationAccess::dataReadyToRetrieve, this, &ShapeUniverse::retrieveAndDisplayData, Qt::QueuedConnection);

	ResolveDescription resolveDesc;
	resolveDesc.resolveCellLinks = true;
	IntRect rect = _viewport->getRect();
	_simAccess->requireData(rect, resolveDesc);
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
	_itemManager->update(_visualDesc);
}

namespace
{
	void collectIds(std::list<QGraphicsItem*> const &items, list<uint64_t> &cellIds, list<uint64_t> &particleIds)
	{
		for (auto item : items) {
			if (auto cellItem = qgraphicsitem_cast<CellItem*>(item)) {
				cellIds.push_back(cellItem->getId());
			}
			if (auto particleItem = qgraphicsitem_cast<ParticleItem*>(item)) {
				particleIds.push_back(particleItem->getId());
			}
		}
	}

	bool clickedOnSpace(std::list<QGraphicsItem*> const &items)
	{
		for (auto item : items) {
			if (qgraphicsitem_cast<CellItem*>(item) || qgraphicsitem_cast<ParticleItem*>(item)) {
				return false;
			}
		}
		return true;
	}
}

void ShapeUniverse::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
	auto itemsClicked = QGraphicsScene::items(e->scenePos()).toStdList();
	list<uint64_t> cellIds;
	list<uint64_t> particleIds;
	collectIds(itemsClicked, cellIds, particleIds);

	if (!_visualDesc->isInSelection(cellIds) || !_visualDesc->isInSelection(particleIds)) {
		_visualDesc->setSelection(cellIds, particleIds);
		_itemManager->update(_visualDesc);
	}

	if (clickedOnSpace(itemsClicked)) {
		_visualDesc->setSelection(list<uint64_t>(), list<uint64_t>());
		auto pos = CoordinateSystem::sceneToModel(e->scenePos());
		_itemManager->setMarkerItem(pos, pos);
		_itemManager->update(_visualDesc);
	}
}

void ShapeUniverse::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
	bool leftButton = ((e->buttons() & Qt::LeftButton) == Qt::LeftButton);
	bool rightButton = ((e->buttons() & Qt::RightButton) == Qt::RightButton);
	
	if(_itemManager->isMarkerActive()) {
		auto pos = CoordinateSystem::sceneToModel(e->scenePos());
		_itemManager->setMarkerLowerRight(pos);
		auto itemsWithinMarker = _itemManager->getItemsWithinMarker();
		list<uint64_t> cellIds;
		list<uint64_t> particleIds;
		collectIds(itemsWithinMarker, cellIds, particleIds);
		_visualDesc->setSelection(cellIds, particleIds);
		_itemManager->update(_visualDesc);
	}
	if (!_itemManager->isMarkerActive()) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		if (leftButton) {
			_visualDesc->moveSelection(delta);
			_itemManager->update(_visualDesc);
		}
		if (rightButton) {
			_visualDesc->moveExtendedSelection(delta);
			_itemManager->update(_visualDesc);
		}
		_visualDesc->setToUnmodified();
	}
}

void ShapeUniverse::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();
	}
}

