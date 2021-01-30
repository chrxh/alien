#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>
#include <QScrollbar>

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
#include "ItemViewport.h"

ItemUniverseView::ItemUniverseView(QGraphicsView* graphicsView, QObject *parent)
	: UniverseView(parent), _graphicsView(graphicsView)
{
    _scene = new QGraphicsScene(parent);
    _scene->setBackgroundBrush(QBrush(Const::UniverseColor));
    _scene->installEventFilter(this);
}


void ItemUniverseView::init(Notifier* notifier, SimulationController* controller, DataRepository* manipulator)
{
    disconnectView();

    _controller = controller;
	_repository = manipulator;
	_notifier = notifier;

    delete _viewport;
    _viewport = new ItemViewport(_graphicsView, this);

	auto itemManager = new ItemManager();
	SET_CHILD(_itemManager, itemManager);

	_itemManager->init(_scene, _viewport, _controller->getContext()->getSimulationParameters());

	connect(_notifier, &Notifier::toggleCellInfo, this, &ItemUniverseView::cellInfoToggled);
}

void ItemUniverseView::connectView()
{
    disconnectView();
    _connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &ItemUniverseView::requestData));
    _connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &ItemUniverseView::receivedNotifications));
    _connections.push_back(QObject::connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &ItemUniverseView::scrolled));
    _connections.push_back(QObject::connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &ItemUniverseView::scrolled));
}

void ItemUniverseView::disconnectView()
{
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void ItemUniverseView::refresh()
{
    requestData();
}

bool ItemUniverseView::isActivated() const
{
    return _graphicsView->scene() == _scene;
}

void ItemUniverseView::activate(double zoomFactor)
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    _graphicsView->setScene(_scene);
    _graphicsView->resetTransform();

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    _itemManager->activate(size);
    setZoomFactor(zoomFactor);
}

double ItemUniverseView::getZoomFactor() const
{
    return _zoomFactor;
}

void ItemUniverseView::setZoomFactor(double zoomFactor)
{
    _zoomFactor = zoomFactor;
    _graphicsView->resetTransform();
    _graphicsView->scale(CoordinateSystem::sceneToModel(_zoomFactor), CoordinateSystem::sceneToModel(_zoomFactor));
}

QVector2D ItemUniverseView::getCenterPositionOfScreen() const
{
    auto const width = static_cast<double>(_graphicsView->width());
    auto const height = static_cast<double>(_graphicsView->height());
    auto const sceneCoordinates = _graphicsView->mapToScene(width / 2.0, height / 2.0);
    auto const modelCoordinates = CoordinateSystem::sceneToModel(QVector2D(sceneCoordinates.x(), sceneCoordinates.y()));
    return modelCoordinates;
}

void ItemUniverseView::centerTo(QVector2D const & position)
{
    auto const scenePosition = CoordinateSystem::modelToScene(position);
    _graphicsView->centerOn(scenePosition.x(), scenePosition.y());
}

/*
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
*/

void ItemUniverseView::toggleCenterSelection(bool value)
{
	_centerSelection = value;
	centerSelectionIfEnabled();
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

void ItemUniverseView::centerSelectionIfEnabled()
{
	if (_centerSelection && !_mouseButtonPressed) {
        if (auto const& centerPos = getCenterPosOfSelection()) {
            disconnectView();
            _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
            auto const scenePos = CoordinateSystem::modelToScene(*centerPos);
            _graphicsView->centerOn(scenePos.x(), scenePos.y());
            _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
            connectView();
        }
	}
}

void ItemUniverseView::updateItems()
{
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _itemManager->update(_repository);
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    _scene->update();
}

void ItemUniverseView::receivedNotifications(set<Receiver> const& targets)
{
	if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}

	centerSelectionIfEnabled();
    updateItems();
}

void ItemUniverseView::cellInfoToggled(bool showInfo)
{
	_itemManager->toggleCellInfo(showInfo);
	if (!_connections.empty()) {
        updateItems();
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
    updateItems();
}

void ItemUniverseView::startMarking(QPointF const& scenePos)
{
	_repository->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
    updateItems();
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

bool ItemUniverseView::eventFilter(QObject * object, QEvent * event)
{
    if (object != _scene) {
        return false;
    }

    if (event->type() == QEvent::GraphicsSceneMousePress) {
        mousePressEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    if (event->type() == QEvent::GraphicsSceneMouseMove) {
        mouseMoveEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    if (event->type() == QEvent::GraphicsSceneMouseRelease) {
        mouseReleaseEvent(static_cast<QGraphicsSceneMouseEvent*>(event));
    }

    return false;
}

void ItemUniverseView::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
	_mouseButtonPressed = true;
	auto itemsClicked = _scene->items(e->scenePos()).toStdList();
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
        updateItems();
    }
	if (!_itemManager->isMarkerActive()) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		if (leftButton && !rightButton) {
			_repository->moveSelection(delta);
			_repository->reconnectSelectedCells();
            updateItems();
        }
		if (rightButton && !leftButton) {
			_repository->moveExtendedSelection(delta);
            updateItems();
        }
		if (leftButton && rightButton) {
			_repository->rotateSelection(delta.y()*10);
            updateItems();
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

