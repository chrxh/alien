#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Base/Definitions.h"
#include "Model/Api/SimulationController.h"
#include "Model/Api/ModelBuilderFacade.h"
#include "Model/Api/SimulationContext.h"
#include "Model/Api/SpaceMetric.h"
#include "Gui/VisualEditor/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/DataManipulator.h"
#include "Gui/Notifier.h"

#include "ShapeUniverse.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "CoordinateSystem.h"

ShapeUniverse::ShapeUniverse(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(UNIVERSE_COLOR));
}

ShapeUniverse::~ShapeUniverse()
{
}

void ShapeUniverse::init(Notifier* notifier, SimulationController * controller, DataManipulator* manipulator, ViewportInterface * viewport)
{
	_controller = controller;
	_viewport = viewport;
	_manipulator = manipulator;
	_notifier = notifier;

	auto itemManager = new ItemManager();
	SET_CHILD(_itemManager, itemManager);

	_itemManager->init(this, viewport, _controller->getContext()->getSimulationParameters());
}

void ShapeUniverse::activate()
{
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_itemManager->activate(size);

	connect(_controller, &SimulationController::nextFrameCalculated, this, &ShapeUniverse::requestData);
	connect(_notifier, &Notifier::notify, this, &ShapeUniverse::receivedNotifications);

	_manipulator->requireDataUpdateFromSimulation(_viewport->getRect());
}

void ShapeUniverse::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &ShapeUniverse::requestData);
	disconnect(_notifier, &Notifier::notify, this, &ShapeUniverse::receivedNotifications);
}

void ShapeUniverse::requestData()
{
	_manipulator->requireDataUpdateFromSimulation(_viewport->getRect());
}

void ShapeUniverse::receivedNotifications(set<Receiver> const& targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}
	_itemManager->update(_manipulator);
}

ShapeUniverse::Selection ShapeUniverse::getSelectionFromItems(std::list<QGraphicsItem*> const &items) const
{
	ShapeUniverse::Selection result;
	for (auto item : items) {
		if (auto cellItem = qgraphicsitem_cast<CellItem*>(item)) {
			result.cellIds.push_back(cellItem->getId());
		}
		if (auto particleItem = qgraphicsitem_cast<ParticleItem*>(item)) {
			result.particleIds.push_back(particleItem->getId());
		}
	}
	return result;
}

void ShapeUniverse::delegateSelection(Selection const & selection)
{
	_manipulator->setSelection(selection.cellIds, selection.particleIds);
	_itemManager->update(_manipulator);
}

void ShapeUniverse::startMarking(QPointF const& scenePos)
{
	_manipulator->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
	_itemManager->update(_manipulator);
}

namespace
{
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
	Selection selection = getSelectionFromItems(itemsClicked);

	if (!_manipulator->isInSelection(selection.cellIds) || !_manipulator->isInSelection(selection.particleIds)) {
		delegateSelection(selection);
	}

	if (clickedOnSpace(itemsClicked)) {
		startMarking(e->scenePos());
	}
	Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::Toolbar }, UpdateDescription::All);
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
		auto selection = getSelectionFromItems(itemsWithinMarker);
		_manipulator->setSelection(selection.cellIds, selection.particleIds);
		_itemManager->update(_manipulator);
	}
	if (!_itemManager->isMarkerActive()) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		if (leftButton) {
			_manipulator->moveSelection(delta);
			_manipulator->reconnectSelectedCells();
			_itemManager->update(_manipulator);
		}
		if (rightButton) {
			_manipulator->moveExtendedSelection(delta);
			_itemManager->update(_manipulator);
		}
	}
	if (leftButton || rightButton) {
		Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::Toolbar }, UpdateDescription::TokenUnchanged);
	}
}

void ShapeUniverse::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();

	}
	else {
		if (_manipulator->areEntitiesSelected()) {
			Q_EMIT _notifier->notify({ Receiver::Simulation }, UpdateDescription::TokenUnchanged);
		}
	}
}

