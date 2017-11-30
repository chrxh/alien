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

#include "ItemUniverseView.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "CoordinateSystem.h"

ItemUniverseView::ItemUniverseView(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(UNIVERSE_COLOR));
}


void ItemUniverseView::init(Notifier* notifier, SimulationController * controller, DataManipulator* manipulator, ViewportInterface * viewport)
{
	_controller = controller;
	_viewport = viewport;
	_manipulator = manipulator;
	_notifier = notifier;

	auto itemManager = new ItemManager();
	SET_CHILD(_itemManager, itemManager);

	_itemManager->init(this, viewport, _controller->getContext()->getSimulationParameters());
	connect(_notifier, &Notifier::toggleCellInfo, this, &ItemUniverseView::cellInfoToggled);
}

void ItemUniverseView::activate()
{
	IntVector2D size = _controller->getContext()->getSpaceMetric()->getSize();
	_itemManager->activate(size);

	connect(_controller, &SimulationController::nextFrameCalculated, this, &ItemUniverseView::requestData);
	connect(_notifier, &Notifier::notify, this, &ItemUniverseView::receivedNotifications);
	connect(_viewport, &ViewportInterface::scrolling, this, &ItemUniverseView::scrolling);

	requestData();
}

void ItemUniverseView::deactivate()
{
	disconnect(_controller, &SimulationController::nextFrameCalculated, this, &ItemUniverseView::requestData);
	disconnect(_notifier, &Notifier::notify, this, &ItemUniverseView::receivedNotifications);
	disconnect(_viewport, &ViewportInterface::scrolling, this, &ItemUniverseView::scrolling);
}

void ItemUniverseView::refresh()
{
	requestData();
}

void ItemUniverseView::requestData()
{
	_manipulator->requireDataUpdateFromSimulation(_viewport->getRect());
}

void ItemUniverseView::receivedNotifications(set<Receiver> const& targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}
	_itemManager->update(_manipulator);
}

void ItemUniverseView::cellInfoToggled(bool showInfo)
{
	_itemManager->toggleCellInfo(showInfo);
	_itemManager->update(_manipulator);
}

void ItemUniverseView::scrolling()
{
	requestData();
}

ItemUniverseView::Selection ItemUniverseView::getSelectionFromItems(std::list<QGraphicsItem*> const &items) const
{
	ItemUniverseView::Selection result;
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

void ItemUniverseView::delegateSelection(Selection const & selection)
{
	_manipulator->setSelection(selection.cellIds, selection.particleIds);
	_itemManager->update(_manipulator);
}

void ItemUniverseView::startMarking(QPointF const& scenePos)
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

void ItemUniverseView::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
	auto itemsClicked = QGraphicsScene::items(e->scenePos()).toStdList();
	Selection selection = getSelectionFromItems(itemsClicked);

	bool alreadySelected = _manipulator->isInSelection(selection.cellIds) && _manipulator->isInSelection(selection.particleIds);
	if (!alreadySelected) {
		delegateSelection(selection);
	}

	if (clickedOnSpace(itemsClicked)) {
		startMarking(e->scenePos());
	}

	if (alreadySelected) {
		Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::Toolbar }, UpdateDescription::AllExceptToken);
	}
	else {
		Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::Toolbar }, UpdateDescription::All);
	}
}

void ItemUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
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
		if (leftButton && !rightButton) {
			_manipulator->moveSelection(delta);
			_manipulator->reconnectSelectedCells();
			_itemManager->update(_manipulator);
		}
		if (rightButton && !leftButton) {
			_manipulator->moveExtendedSelection(delta);
			_itemManager->update(_manipulator);
		}
		if (leftButton && rightButton) {
			_manipulator->rotateSelection(delta.y()*10);
			_itemManager->update(_manipulator);
		}
	}
	if (leftButton || rightButton) {
		Q_EMIT _notifier->notify({ Receiver::DataEditor, Receiver::Toolbar }, UpdateDescription::AllExceptToken);
	}
}

void ItemUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();

	}
	else {
		if (_manipulator->areEntitiesSelected()) {
			Q_EMIT _notifier->notify({ Receiver::Simulation }, UpdateDescription::AllExceptToken);
		}
	}
}

