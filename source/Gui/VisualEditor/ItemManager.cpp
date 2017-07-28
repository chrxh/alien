#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "ItemManager.h"
#include "VisualDescription.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "CellConnectionItem.h"
#include "CoordinateSystem.h"
#include "MarkerItem.h"

void ItemManager::init(QGraphicsScene * scene, ViewportInterface* viewport, SimulationParameters* parameters)
{
	auto config = new ItemConfig();

	_scene = scene;
	_viewport = viewport;
	_parameters = parameters;
	SET_CHILD(_config, config);

	_config->init(parameters);
}

void ItemManager::activate(IntVector2D size)
{
	_scene->clear();
	_scene->setSceneRect(0, 0, CoordinateSystem::modelToScene(size.x), CoordinateSystem::modelToScene(size.y));
	_cellsByIds.clear();
	_particlesByIds.clear();
}

void ItemManager::updateCells(VisualDescription* visualDesc)
{
	auto const &data = visualDesc->getDataRef();

	map<uint64_t, CellItem*> newCellsByIds;
	for (auto const &clusterT : data.clusters) {
		auto const &cluster = clusterT.getValue();
		for (auto const &cellT : cluster.cells) {
			auto const &cell = cellT.getValue();
			auto it = _cellsByIds.find(cell.id);
			CellItem* item;
			if (it != _cellsByIds.end()) {
				item = it->second;
				item->update(cell);
				newCellsByIds[cell.id] = item;
				_cellsByIds.erase(it);
			}
			else {
				item = new CellItem(_config, cell);
				_scene->addItem(item);
				newCellsByIds[cell.id] = item;
			}
			if (visualDesc->isInSelection(cell.id)) {
				item->setFocusState(CellItem::FOCUS_CELL);
			}
			else if (visualDesc->isInExtendedSelection(cell.id)) {
				item->setFocusState(CellItem::FOCUS_CLUSTER);
			}
			else {
				item->setFocusState(CellItem::NO_FOCUS);
			}
		}
	}
	for (auto const& cellById : _cellsByIds) {
		delete cellById.second;
	}
	_cellsByIds = newCellsByIds;
}

void ItemManager::updateParticles(VisualDescription* visualDesc)
{
	auto const &data = visualDesc->getDataRef();

	map<uint64_t, ParticleItem*> newParticlesByIds;
	for (auto const &particleT : data.particles) {
		auto const &particle = particleT.getValue();
		auto it = _particlesByIds.find(particle.id);
		ParticleItem* item;
		if (it != _particlesByIds.end()) {
			item = it->second;
			item->update(particle);
			newParticlesByIds[particle.id] = item;
			_particlesByIds.erase(it);
		}
		else {
			item = new ParticleItem(_config, particle);
			_scene->addItem(item);
			newParticlesByIds[particle.id] = item;
		}
		if (visualDesc->isInSelection(particle.id)) {
			item->setFocusState(ParticleItem::FOCUS);
		}
		else {
			item->setFocusState(ParticleItem::NO_FOCUS);
		}
	}
	for (auto const& particleById : _particlesByIds) {
		delete particleById.second;
	}
	_particlesByIds = newParticlesByIds;
}

void ItemManager::updateConnections(VisualDescription* visualDesc)
{
	auto const &data = visualDesc->getDataRef();

	map<set<uint64_t>, CellConnectionItem*> newConnectionsByIds;
	for (auto const &clusterT : data.clusters) {
		auto const &clusterD = clusterT.getValue();
		for (auto const &cellT : clusterD.cells) {
			auto const &cellD = cellT.getValue();
			if (!cellD.connectingCells.isInitialized()) {
				continue;
			}
			for (uint64_t connectingCellId : cellD.connectingCells.getValue()) {
				auto &connectingCellD = visualDesc->getCellDescRef(connectingCellId);
				set<uint64_t> connectionId;
				connectionId.insert(cellD.id);
				connectionId.insert(connectingCellId);
				if (newConnectionsByIds.find(connectionId) != newConnectionsByIds.end()) {
					continue;
				}
				auto connectionIt = _connectionsByIds.find(connectionId);
				if (connectionIt != _connectionsByIds.end()) {
					CellConnectionItem* connection = connectionIt->second;
					connection->update(cellD, connectingCellD);
					newConnectionsByIds[connectionId] = connection;
					_connectionsByIds.erase(connectionIt);
				}
				else {
					CellConnectionItem* newConnection = new CellConnectionItem(_config, cellD, connectingCellD);
					_scene->addItem(newConnection);
					newConnectionsByIds[connectionId] = newConnection;
				}
			}
		}
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_connectionsByIds = newConnectionsByIds;
}

void ItemManager::update(VisualDescription* visualDesc)
{
	_viewport->setModeToNoUpdate();

	updateCells(visualDesc);
	updateConnections(visualDesc);
	updateParticles(visualDesc);

	_viewport->setModeToUpdate();
	_scene->update();
}

void ItemManager::setMarkerItem(QPointF const &upperLeft, QPointF const &lowerRight)
{
	if (_marker) {
		_marker->update(upperLeft, lowerRight);
	}
	else {
		_marker = new MarkerItem(upperLeft, lowerRight);
		_scene->addItem(_marker);
	}
	_scene->update();
}

void ItemManager::setMarkerLowerRight(QPointF const & lowerRight)
{
	if (_marker) {
		_marker->setLowerRight(lowerRight);
		_scene->update();
	}
}

void ItemManager::deleteMarker()
{
	if (_marker) {
		delete _marker;
		_marker = nullptr;
		_scene->update();
	}
}

bool ItemManager::isMarkerActive() const
{
	return (bool)_marker;
}

std::list<QGraphicsItem*> ItemManager::getItemsWithinMarker() const
{
	return _marker->collidingItems().toStdList();
}
