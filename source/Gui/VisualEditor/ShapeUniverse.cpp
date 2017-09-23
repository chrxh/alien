#include <QGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "Base/Definitions.h"
#include "Gui/Settings.h"
#include "Gui/VisualEditor/ViewportInterface.h"
#include "Gui/DataEditor/DataEditorModel.h"
#include "Gui/DataEditor/DataEditorContext.h"
#include "Model/SimulationController.h"
#include "Model/ModelBuilderFacade.h"
#include "Model/CellConnector.h"
#include "Model/AccessPorts/SimulationAccess.h"
#include "Model/Context/SimulationContextApi.h"
#include "Model/Context/SpaceMetricApi.h"

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

void ShapeUniverse::init(SimulationController * controller, SimulationAccess* access, ViewportInterface * viewport
	, DataEditorContext* dataEditorContext)
{
	ModelBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBuilderFacade>();
	_controller = controller;
	_viewport = viewport;
	_simAccess = access;
	_dataEditorContext = dataEditorContext;

	auto items = new ItemManager();
	auto descManager = new VisualDescription();
	auto connector = facade->buildCellConnector(controller->getContext());
	SET_CHILD(_connector, connector);
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
	_visualDesc->setSelection(selection.cellIds, selection.particleIds);
	_itemManager->update(_visualDesc);

	auto dataEditorModel = _dataEditorContext->getModel();
	dataEditorModel->selectedCellIds = selection.cellIds;
	dataEditorModel->selectedParticleIds = selection.particleIds;
	_dataEditorContext->notifyDataEditor();
}

void ShapeUniverse::startMarking(QPointF const& scenePos)
{
	_visualDesc->setSelection(list<uint64_t>(), list<uint64_t>());
	auto pos = CoordinateSystem::sceneToModel(scenePos);
	_itemManager->setMarkerItem(pos, pos);
	_itemManager->update(_visualDesc);

	auto dataEditorModel = _dataEditorContext->getModel();
	dataEditorModel->selectedCellIds.clear();
	dataEditorModel->selectedParticleIds.clear();
	_dataEditorContext->notifyDataEditor();
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

	if (!_visualDesc->isInSelection(selection.cellIds) || !_visualDesc->isInSelection(selection.particleIds)) {
		delegateSelection(selection);
	}

	if (clickedOnSpace(itemsClicked)) {
		startMarking(e->scenePos());
	}
	_savedDataBeforeMovement = _visualDesc->getDataRef();
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
		_visualDesc->setSelection(selection.cellIds, selection.particleIds);
		_itemManager->update(_visualDesc);
	}
	if (!_itemManager->isMarkerActive()) {
		auto lastPos = e->lastScenePos();
		auto pos = e->scenePos();
		QVector2D delta(pos.x() - lastPos.x(), pos.y() - lastPos.y());
		delta = CoordinateSystem::sceneToModel(delta);
		if (leftButton) {
			_visualDesc->moveSelection(delta);
			_connector->reconnect(_visualDesc->getDataRef(), _visualDesc->getSelectedCellIds());
			_visualDesc->updateAfterCellReconnections();
			_itemManager->update(_visualDesc);
		}
		if (rightButton) {
			_visualDesc->moveExtendedSelection(delta);
			_itemManager->update(_visualDesc);
		}
	}
}

void ShapeUniverse::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
	if (_itemManager->isMarkerActive()) {
		_itemManager->deleteMarker();

	}
	else {
		if (_visualDesc->areEntitiesSelected()) {
			DataChangeDescription delta(_savedDataBeforeMovement, _visualDesc->getDataRef());
			_simAccess->updateData(delta);
		}
	}
}

