/*
#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Base/Definitions.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "Gui/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "ViewportController.h"
#include "ItemUniverseView.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "CoordinateSystem.h"

ItemUniverseView::ItemUniverseView(QObject *parent)
	: QGraphicsScene(parent)
{
    setBackgroundBrush(QBrush(Const::UniverseColor));
}


void ItemUniverseView::init(Notifier* notifier, SimulationController * controller, DataRepository* manipulator, ViewportInterface * viewport)
{
    _controller = controller;
	_viewport = viewport;
	_repository = manipulator;
	_notifier = notifier;

	auto itemManager = new ItemManager();
	SET_CHILD(_itemManager, itemManager);

	_itemManager->init(this, viewport, _controller->getContext()->getSimulationParameters());

	connect(_notifier, &Notifier::toggleCellInfo, this, &ItemUniverseView::cellInfoToggled);
}

void ItemUniverseView::activate()
{
	IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
	_itemManager->activate(size);

	_connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &ItemUniverseView::requestData));
	_connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &ItemUniverseView::receivedNotifications));
	_connections.push_back(connect(_viewport, &ViewportInterface::scrolled, this, &ItemUniverseView::scrolled));
    _connections.push_back(connect(_viewport, &ViewportInterface::zoomed, this, &ItemUniverseView::scrolled));

	requestData();
	_activated = true;
}

void ItemUniverseView::deactivate()
{
	for (auto const& connection : _connections) {
		disconnect(connection);
	}
	_activated = false;
}

void ItemUniverseView::refresh()
{
	requestData();
}

void ItemUniverseView::toggleCenterSelection(bool value)
{
	_centerSelection = value;
	centerSelectionIfEnabled(NotifyScrollChanged::Yes);
}

void ItemUniverseView::requestData()
{
	_repository->requireDataUpdateFromSimulation(_viewport->getRect());
}

optional<QVector2D> ItemUniverseView::getCenterPosOfSelection() const
{
	QVector2D result;
	int numEntities = 0;
	for (auto selectedCellId : _repository->getSelectedCellIds()) {
		auto const& cell = _repository->getCellDescRef(selectedCellId);
		result += *cell.pos;
		++numEntities;
	}
	for (auto selectedParticleId : _repository->getSelectedParticleIds()) {
		auto const& particle = _repository->getParticleDescRef(selectedParticleId);
		result += *particle.pos;
		++numEntities;
	}
	if (numEntities == 0) {
		return boost::none;
	}
	result /= numEntities;
	return result;
}

void ItemUniverseView::centerSelectionIfEnabled(NotifyScrollChanged notify)
{
	if (_centerSelection && !_mouseButtonPressed) {
		if (auto const& centerPos = getCenterPosOfSelection()) {
			_viewport->setModeToNoUpdate();
			_viewport->scrollToPos(*centerPos, notify);
			_viewport->setModeToNoUpdate();
		}
	}
}

void ItemUniverseView::receivedNotifications(set<Receiver> const& targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}
	centerSelectionIfEnabled(NotifyScrollChanged::No);
	_itemManager->update(_repository);
}

void ItemUniverseView::cellInfoToggled(bool showInfo)
{
	_itemManager->toggleCellInfo(showInfo);
	if (_activated) {
		_itemManager->update(_repository);
	}
}

void ItemUniverseView::scrolled()
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
	_repository->setSelection(selection.cellIds, selection.particleIds);
	_itemManager->update(_repository);
}

void ItemUniverseView::startMarking(QPointF const& scenePos)
{
	_repository->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
	_itemManager->update(_repository);
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
	_mouseButtonPressed = true;
	auto itemsClicked = QGraphicsScene::items(e->scenePos()).toStdList();
	list<QGraphicsItem*> frontItem = !itemsClicked.empty() ? list<QGraphicsItem*>({ itemsClicked.front() }) : list<QGraphicsItem*>();
	Selection selection = getSelectionFromItems(frontItem);

	bool alreadySelected = _repository->isInSelection(selection.cellIds) && _repository->isInSelection(selection.particleIds);
	if (!alreadySelected) {
		delegateSelection(selection);
	}

	if (clickedOnSpace(itemsClicked)) {
		startMarking(e->scenePos());
	}

	if (alreadySelected) {
		Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::DataEditor, Receiver::ActionController }, UpdateDescription::AllExceptToken);
	}
	else {
		Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::DataEditor, Receiver::ActionController }, UpdateDescription::All);
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
		if (!selection.particleIds.empty()) {
			int dummy = 0;
		}
		_repository->setSelection(selection.cellIds, selection.particleIds);
		_itemManager->update(_repository);
	}
	if (!_itemManager->isMarkerActive()) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		if (leftButton && !rightButton) {
			_repository->moveSelection(delta);
			_repository->reconnectSelectedCells();
			_itemManager->update(_repository);
		}
		if (rightButton && !leftButton) {
			_repository->moveExtendedSelection(delta);
			_itemManager->update(_repository);
		}
		if (leftButton && rightButton) {
			_repository->rotateSelection(delta.y()*10);
			_itemManager->update(_repository);
		}
	}
	if (leftButton || rightButton) {
		Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::DataEditor, Receiver::ActionController }, UpdateDescription::AllExceptToken);
	}
}

void ItemUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
	_mouseButtonPressed = false;
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();

	}
	else {
		if (_repository->areEntitiesSelected()) {
			Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation }, UpdateDescription::AllExceptToken);
		}
	}
}

*/
