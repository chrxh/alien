#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>
#include <QScrollbar>

#include "Base/ServiceLocator.h"
#include "Base/Definitions.h"
#include "Base/DebugMacros.h"

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationContext.h"
#include "EngineInterface/SpaceProperties.h"
#include "Gui/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"

#include "ItemUniverseView.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "CoordinateSystem.h"
#include "ItemViewport.h"

ItemUniverseView::ItemUniverseView(QGraphicsView* graphicsView, QObject *parent)
    : UniverseView(graphicsView, parent)
{
    _scene = new QGraphicsScene(parent);
    _scene->setBackgroundBrush(QBrush(Const::UniverseColor));
    _scene->installEventFilter(this);
}


void ItemUniverseView::init(Notifier* notifier, SimulationController* controller, DataRepository* manipulator)
{
    TRY;
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
    CATCH;
}

void ItemUniverseView::connectView()
{
    TRY;
    disconnectView();
    _connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &ItemUniverseView::requestData));
    _connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &ItemUniverseView::receivedNotifications));
    _connections.push_back(QObject::connect(_graphicsView->horizontalScrollBar(), &QScrollBar::valueChanged, this, &ItemUniverseView::scrolled));
    _connections.push_back(QObject::connect(_graphicsView->verticalScrollBar(), &QScrollBar::valueChanged, this, &ItemUniverseView::scrolled));
    CATCH;
}

void ItemUniverseView::disconnectView()
{
    TRY;
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
    CATCH;
}

void ItemUniverseView::refresh()
{
    TRY;
    requestData();
    CATCH;
}

bool ItemUniverseView::isActivated() const
{
    TRY;
    return _graphicsView->scene() == _scene;
    CATCH;
}

void ItemUniverseView::activate(double zoomFactor)
{
    TRY;
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    _graphicsView->setScene(_scene);
    _graphicsView->resetTransform();

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    _itemManager->activate(size);
    setZoomFactor(zoomFactor);
    CATCH;
}

double ItemUniverseView::getZoomFactor() const
{
    TRY;
    return _zoomFactor;
    CATCH;
}

void ItemUniverseView::setZoomFactor(double zoomFactor)
{
    TRY;
    _zoomFactor = zoomFactor;
    _graphicsView->resetTransform();
    _graphicsView->scale(CoordinateSystem::sceneToModel(_zoomFactor), CoordinateSystem::sceneToModel(_zoomFactor));
    CATCH;
}

void ItemUniverseView::setZoomFactor(double zoomFactor, QVector2D const& fixedPos)
{

}


QVector2D ItemUniverseView::getCenterPositionOfScreen() const
{
    TRY;
    auto const width = static_cast<double>(_graphicsView->width());
    auto const height = static_cast<double>(_graphicsView->height());
    auto const sceneCoordinates = _graphicsView->mapToScene(width / 2.0, height / 2.0);
    auto const modelCoordinates = CoordinateSystem::sceneToModel(QVector2D(sceneCoordinates.x(), sceneCoordinates.y()));
    return modelCoordinates;
    CATCH;
}

void ItemUniverseView::centerTo(QVector2D const & position)
{
    TRY;
    centerToIntern(CoordinateSystem::modelToScene(position));
    CATCH;
}

void ItemUniverseView::toggleCenterSelection(bool value)
{
    TRY;
    _centerSelection = value;
	centerSelectionIfEnabled();
    CATCH;
}

void ItemUniverseView::requestData()
{
    TRY;
    _repository->requireDataUpdateFromSimulation(_viewport->getRect());
    CATCH;
}

boost::optional<QVector2D> ItemUniverseView::getCenterPosOfSelection() const
{
    TRY;
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
    CATCH;
}

void ItemUniverseView::centerSelectionIfEnabled()
{
    TRY;
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
    CATCH;
}

void ItemUniverseView::updateItems()
{
    TRY;
    _graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _itemManager->update(_repository);
    _graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    _scene->update();
    CATCH;
}

void ItemUniverseView::receivedNotifications(set<Receiver> const& targets)
{
    TRY;
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}

	centerSelectionIfEnabled();
    updateItems();
    CATCH;
}

void ItemUniverseView::cellInfoToggled(bool showInfo)
{
    TRY;
    _itemManager->toggleCellInfo(showInfo);
	if (!_connections.empty()) {
        updateItems();
	}
    CATCH;
}

void ItemUniverseView::scrolled()
{
    TRY;
    requestData();
    CATCH;
}

ItemUniverseView::Selection ItemUniverseView::getSelectionFromItems(QList<QGraphicsItem*> const &items) const
{
    TRY;
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
    CATCH;
}

void ItemUniverseView::delegateSelection(Selection const & selection)
{
    TRY;
    _repository->setSelection(selection.cellIds, selection.particleIds);
    updateItems();
    CATCH;
}

void ItemUniverseView::startMarking(QPointF const& scenePos)
{
    TRY;
    _repository->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
    updateItems();
    CATCH;
}

namespace
{
	bool clickedOnSpace(QList<QGraphicsItem*> const &items)
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
    TRY;
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
    CATCH;
}

void ItemUniverseView::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
    TRY;
    auto pos = QVector2D(CoordinateSystem::sceneToModel(e->scenePos().x()), CoordinateSystem::sceneToModel(e->scenePos().y()));
    if (e->buttons() == Qt::MouseButton::LeftButton) {
        Q_EMIT startContinuousZoomIn(pos);
    }
    if (e->buttons() == Qt::MouseButton::RightButton) {
        Q_EMIT startContinuousZoomOut(pos);
    }
/*
    _mouseButtonPressed = true;
	auto itemsClicked = _scene->items(e->scenePos());
	QList<QGraphicsItem*> frontItem = !itemsClicked.empty() ? QList<QGraphicsItem*>({ itemsClicked.front() }) : QList<QGraphicsItem*>();
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
*/
    CATCH;
}

void ItemUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    TRY;
/*
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
*/
    CATCH;
}

void ItemUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
    TRY;
    Q_EMIT endContinuousZoom();
/*
    _mouseButtonPressed = false;
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();

	}
	else {
		if (_repository->areEntitiesSelected()) {
			Q_EMIT _notifier->notifyDataRepositoryChanged({ Receiver::Simulation }, UpdateDescription::AllExceptToken);
		}
	}
*/
    CATCH;
}

