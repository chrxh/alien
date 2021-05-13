#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>
#include <QScrollbar>
#include <QResizeEvent>

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

#include "ItemWorldController.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "ItemManager.h"
#include "CoordinateSystem.h"
#include "ItemViewport.h"
#include "SimulationViewWidget.h"

ItemWorldController::ItemWorldController(SimulationViewWidget* simulationViewWidget, QObject* parent)
    : AbstractWorldController(simulationViewWidget, parent)
{
    _scene = new QGraphicsScene(parent);
    _scene->setBackgroundBrush(QBrush(Const::UniverseColor));
    _scene->installEventFilter(this);
}


void ItemWorldController::init(Notifier* notifier, SimulationController* controller, DataRepository* manipulator)
{
    TRY;
    disconnectView();

    _controller = controller;
	_repository = manipulator;
	_notifier = notifier;

    delete _viewport;
    _viewport = new ItemViewport(_simulationViewWidget->getGraphicsView(), this);

	auto itemManager = new ItemManager();
	SET_CHILD(_itemManager, itemManager);

	_itemManager->init(_scene, _viewport, _controller->getContext()->getSimulationParameters());

	connect(_notifier, &Notifier::toggleCellInfo, this, &ItemWorldController::cellInfoToggled);
    CATCH;
}

void ItemWorldController::setSettings(SimulationViewSettings const& settings)
{
    _settings = settings;
}

void ItemWorldController::connectView()
{
    TRY;
    disconnectView();
    _connections.push_back(
        connect(_controller, &SimulationController::nextFrameCalculated, this, &ItemWorldController::requestData));
    _connections.push_back(
        connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &ItemWorldController::receivedNotifications));

    _connections.push_back(QObject::connect(
        _simulationViewWidget, &SimulationViewWidget::scrolledX, this, &ItemWorldController::scrolledX));
    _connections.push_back(QObject::connect(
        _simulationViewWidget, &SimulationViewWidget::scrolledY, this, &ItemWorldController::scrolledY));
    CATCH;
}

void ItemWorldController::disconnectView()
{
    TRY;
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
    CATCH;
}

void ItemWorldController::refresh()
{
    TRY;
    requestData();
    CATCH;
}

bool ItemWorldController::isActivated() const
{
    TRY;
    return _simulationViewWidget->getGraphicsView()->scene() == _scene;
    CATCH;
}

void ItemWorldController::activate(double zoomFactor)
{
    TRY;
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    graphicsView->setScene(_scene);
    graphicsView->resetTransform();

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    _itemManager->activate(size);
    setZoomFactor(zoomFactor);
    CATCH;
}

double ItemWorldController::getZoomFactor() const
{
    TRY;
    return _zoomFactor;
    CATCH;
}

void ItemWorldController::setZoomFactor(double zoomFactor)
{
    TRY;
    _zoomFactor = zoomFactor;
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    graphicsView->resetTransform();
    graphicsView->scale(CoordinateSystem::sceneToModel(_zoomFactor), CoordinateSystem::sceneToModel(_zoomFactor));
    updateScrollbars();
    CATCH;
}

void ItemWorldController::setZoomFactor(double zoomFactor, IntVector2D const& viewPos)
{
    TRY;
    auto scenePos = _simulationViewWidget->getGraphicsView()->mapToScene(viewPos.x, viewPos.y);
    auto worldPos = CoordinateSystem::sceneToModel(scenePos);
    setZoomFactor(zoomFactor);
    centerTo(QVector2D(worldPos.x(), worldPos.y()), viewPos);
    CATCH;
}


QVector2D ItemWorldController::getCenterPositionOfScreen() const
{
    TRY;
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    auto const width = static_cast<double>(graphicsView->width());
    auto const height = static_cast<double>(graphicsView->height());
    auto const sceneCoordinates = graphicsView->mapToScene(width / 2.0, height / 2.0);
    auto const modelCoordinates = CoordinateSystem::sceneToModel(QVector2D(sceneCoordinates.x(), sceneCoordinates.y()));
    return modelCoordinates;
    CATCH;
}

void ItemWorldController::centerTo(QVector2D const & position)
{
    TRY;
    auto scenePos = CoordinateSystem::modelToScene(position);
    _simulationViewWidget->getGraphicsView()->centerOn(scenePos.x(), scenePos.y());
    updateScrollbars();
    CATCH;
}

void ItemWorldController::toggleCenterSelection(bool value)
{
    TRY;
    _centerSelection = value;
	centerSelectionIfEnabled();
    CATCH;
}

void ItemWorldController::centerTo(QVector2D const& worldPosition, IntVector2D const& viewPos)
{
    TRY;
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    auto scenePos = graphicsView->mapToScene(viewPos.x, viewPos.y);
    auto centerScenePos = graphicsView->mapToScene(
        static_cast<float>(graphicsView->width()) / 2.0f, static_cast<float>(graphicsView->height()) / 2.0f);

    QVector2D deltaWorldPos(CoordinateSystem::sceneToModel(scenePos.x() - centerScenePos.x()),
        CoordinateSystem::sceneToModel(scenePos.y() - centerScenePos.y()));

    centerTo(worldPosition - deltaWorldPos);
    CATCH;
}

void ItemWorldController::modifySelection(QPointF const& scenePos)
{
    TRY;
    auto itemsClicked = _scene->items(scenePos);
    auto newSelectedCellIds = getSelectionFromItems(itemsClicked).cellIds;
    auto selectedCellIds = _repository->getSelectedCellIds();
    for (auto const& newSelectedCellId : newSelectedCellIds) {
        if (selectedCellIds.erase(newSelectedCellId) == 0) {
            selectedCellIds.insert(newSelectedCellId);
        }
    }
    auto selectedParticleIds = _repository->getSelectedParticleIds();

    std::list<uint64_t> selectedCellIdList(selectedCellIds.begin(), selectedCellIds.end());
    std::list<uint64_t> selectedParticleIdList(selectedParticleIds.begin(), selectedParticleIds.end());
    _repository->setSelection(selectedCellIdList, selectedParticleIdList);

    updateItems();

    Q_EMIT _notifier->notifyDataRepositoryChanged(
        {Receiver::DataEditor, Receiver::ActionController}, UpdateDescription::AllExceptToken);
    CATCH;
}

namespace
{
    bool clickedOnSpace(QList<QGraphicsItem*> const& items)
    {
        for (auto item : items) {
            if (qgraphicsitem_cast<CellItem*>(item) || qgraphicsitem_cast<ParticleItem*>(item)) {
                return false;
            }
        }
        return true;
    }
}

void ItemWorldController::startNewSelection(QPointF const& scenePos)
{
    TRY;
    auto itemsClicked = _scene->items(scenePos);
    QList<QGraphicsItem*> frontItem =
        !itemsClicked.empty() ? QList<QGraphicsItem*>({itemsClicked.front()}) : QList<QGraphicsItem*>();
    Selection selection = getSelectionFromItems(frontItem);

    bool alreadySelected =
        _repository->isInSelection(selection.cellIds) && _repository->isInSelection(selection.particleIds);
    if (!alreadySelected) {
        delegateSelection(selection);
    }

    if (clickedOnSpace(itemsClicked)) {
        startMarking(scenePos);
    }

    if (alreadySelected) {
        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::DataEditor, Receiver::ActionController}, UpdateDescription::AllExceptToken);
    } else {
        Q_EMIT _notifier->notifyDataRepositoryChanged(
            {Receiver::DataEditor, Receiver::ActionController}, UpdateDescription::All);
    }
    CATCH;
}

void ItemWorldController::updateScrollbars()
{
    _simulationViewWidget->updateScrollbars(
        _controller->getContext()->getSpaceProperties()->getSize(), getCenterPositionOfScreen(), getZoomFactor());
}

void ItemWorldController::resize(QResizeEvent* event)
{
    TRY;
    updateScrollbars();
    CATCH;
}

void ItemWorldController::requestData()
{
    TRY;
    _repository->requireDataUpdateFromSimulation(_viewport->getRect());
    CATCH;
}

boost::optional<QVector2D> ItemWorldController::getCenterPosOfSelection() const
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

void ItemWorldController::centerSelectionIfEnabled()
{
    TRY;
    if (_centerSelection && !_mouseButtonPressed) {
        if (auto const& centerPos = getCenterPosOfSelection()) {
            disconnectView();
            auto graphicsView = _simulationViewWidget->getGraphicsView();
            graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
            auto const scenePos = CoordinateSystem::modelToScene(*centerPos);
            graphicsView->centerOn(scenePos.x(), scenePos.y());
            graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
            connectView();
        }
	}
    CATCH;
}

void ItemWorldController::updateItems()
{
    TRY;
    auto graphicsView = _simulationViewWidget->getGraphicsView();
    graphicsView->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    _itemManager->update(_repository);
    graphicsView->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    _scene->update();
    CATCH;
}

void ItemWorldController::receivedNotifications(set<Receiver> const& targets)
{
    TRY;
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
		return;
	}

	centerSelectionIfEnabled();
    updateItems();
    CATCH;
}

void ItemWorldController::cellInfoToggled(bool showInfo)
{
    TRY;
    _itemManager->toggleCellInfo(showInfo);
	if (!_connections.empty()) {
        updateItems();
	}
    CATCH;
}

void ItemWorldController::scrolledX(float centerX)
{
    TRY;
    auto center = getCenterPositionOfScreen();
    center.setX(centerX);
    centerTo(center);
    requestData();
    CATCH;
}
void ItemWorldController::scrolledY(float centerY)
{
    TRY;
    auto center = getCenterPositionOfScreen();
    center.setY(centerY);
    centerTo(center);
    requestData();
    CATCH;
}

ItemWorldController::Selection ItemWorldController::getSelectionFromItems(QList<QGraphicsItem*> const &items) const
{
    TRY;
    ItemWorldController::Selection result;
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

void ItemWorldController::delegateSelection(Selection const & selection)
{
    TRY;
    _repository->setSelection(selection.cellIds, selection.particleIds);
    updateItems();
    CATCH;
}

void ItemWorldController::startMarking(QPointF const& scenePos)
{
    TRY;
    _repository->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
    updateItems();
    CATCH;
}

bool ItemWorldController::eventFilter(QObject * object, QEvent * event)
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

    if (object = _simulationViewWidget->getGraphicsView()) {
        if (event->type() == QEvent::Resize) {
            resize(static_cast<QResizeEvent*>(event));
        }
    }

    return false;
    CATCH;
}

void ItemWorldController::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    TRY;
    if (SimulationViewSettings::Mode::NavigationMode == _settings.mode) {
        auto viewPos =
            _simulationViewWidget->getGraphicsView()->mapFromScene(event->scenePos().x(), event->scenePos().y());
        auto viewPosInt = IntVector2D{static_cast<int>(viewPos.x()), static_cast<int>(viewPos.y())};
        auto worldPos = CoordinateSystem::sceneToModel(QVector2D(event->scenePos().x(), event->scenePos().y()));

        if (event->buttons() == Qt::MouseButton::LeftButton) {
            Q_EMIT startContinuousZoomIn(viewPosInt);
        }
        if (event->buttons() == Qt::MouseButton::RightButton) {
            Q_EMIT startContinuousZoomOut(viewPosInt);
        }
        if (event->buttons() == Qt::MouseButton::MiddleButton) {
            _worldPosForMovement = worldPos;
        }
    }
    if (SimulationViewSettings::Mode::ActionMode == _settings.mode) {
        if (Qt::KeyboardModifier::ControlModifier == event->modifiers()) {
            modifySelection(event->scenePos());
        } else {
            _mouseButtonPressed = true;
            startNewSelection(event->scenePos());
        }
    }
    CATCH;
}

void ItemWorldController::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    TRY;
    if (SimulationViewSettings::Mode::NavigationMode == _settings.mode) {
        auto viewPos =
            _simulationViewWidget->getGraphicsView()->mapFromScene(event->scenePos().x(), event->scenePos().y());
        auto viewPosInt = IntVector2D{toInt(viewPos.x()), toInt(viewPos.y())};
        if (event->buttons() == Qt::MouseButton::LeftButton) {
            Q_EMIT startContinuousZoomIn(viewPosInt);
        }
        if (event->buttons() == Qt::MouseButton::RightButton) {
            Q_EMIT startContinuousZoomOut(viewPosInt);
        }
        if (event->buttons() == Qt::MouseButton::MiddleButton) {
            centerTo(*_worldPosForMovement, viewPosInt);
            refresh();
        }
    }
    if (SimulationViewSettings::Mode::ActionMode == _settings.mode) {
        bool leftButton = ((event->buttons() & Qt::LeftButton) == Qt::LeftButton);
        bool rightButton = ((event->buttons() & Qt::RightButton) == Qt::RightButton);

        if (_itemManager->isMarkerActive()) {
            auto pos = CoordinateSystem::sceneToModel(event->scenePos());
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
            auto lastPos = event->lastScenePos();
            auto pos = event->scenePos();
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
                _repository->rotateSelection(delta.y() * 10);
                updateItems();
            }
        }
        if (leftButton || rightButton) {
            Q_EMIT _notifier->notifyDataRepositoryChanged(
                {Receiver::DataEditor, Receiver::ActionController}, UpdateDescription::AllExceptToken);
        }
    }
    CATCH;
}

void ItemWorldController::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
    TRY;
    if (SimulationViewSettings::Mode::NavigationMode == _settings.mode) {
        _worldPosForMovement = boost::none;
        Q_EMIT endContinuousZoom();
    }
    if (SimulationViewSettings::Mode::ActionMode == _settings.mode) {
        _mouseButtonPressed = false;
        if (_itemManager->isMarkerActive()) {
            _itemManager->deleteMarker();

        } else {
            if (_repository->areEntitiesSelected()) {
                Q_EMIT _notifier->notifyDataRepositoryChanged(
                    {Receiver::Simulation}, UpdateDescription::AllExceptToken);
            }
        }
    }
    CATCH;
}

