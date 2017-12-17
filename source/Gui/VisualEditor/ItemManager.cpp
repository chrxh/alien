#include <QGraphicsScene>

#include "Model/Api/ChangeDescriptions.h"
#include "Gui/Settings.h"
#include "Gui/DataController.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "ItemManager.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "CellConnectionItem.h"
#include "CoordinateSystem.h"
#include "MarkerItem.h"

void ItemManager::init(QGraphicsScene * scene, ViewportInterface* viewport, SimulationParameters const* parameters)
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
	_connectionsByIds.clear();
}

void ItemManager::updateCells(DataController* manipulator)
{
	auto const &data = manipulator->getDataRef();
	if (!data.clusters) {
		return;
	}

	map<uint64_t, CellItem*> newCellsByIds;
	for (auto const &cluster : *data.clusters) {
		for (auto const &cell : *cluster.cells) {
			auto it = _cellsByIds.find(cell.id);
			CellItem* item;
			if (it != _cellsByIds.end()) {
				item = it->second;
				item->update(cell);
				newCellsByIds.insert_or_assign(cell.id, item);
				_cellsByIds.erase(it);
			}
			else {
				item = new CellItem(_config, cell);
				_scene->addItem(item);
				newCellsByIds[cell.id] = item;
			}
			if (manipulator->isInSelection(cell.id)) {
				item->setFocusState(CellItem::FOCUS_CELL);
			}
			else if (manipulator->isInExtendedSelection(cell.id)) {
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

void ItemManager::updateParticles(DataController* manipulator)
{
	auto const &data = manipulator->getDataRef();
	if (!data.particles) {
		return;
	}

	map<uint64_t, ParticleItem*> newParticlesByIds;
	for (auto const &particle : *data.particles) {
		auto it = _particlesByIds.find(particle.id);
		ParticleItem* item;
		if (it != _particlesByIds.end()) {
			item = it->second;
			item->update(particle);
			newParticlesByIds.insert_or_assign(particle.id, item);
			_particlesByIds.erase(it);
		}
		else {
			item = new ParticleItem(_config, particle);
			_scene->addItem(item);
			newParticlesByIds.insert_or_assign(particle.id, item);
		}
		if (manipulator->isInSelection(particle.id)) {
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

void ItemManager::updateConnections(DataController* visualDesc)
{
	auto const &data = visualDesc->getDataRef();
	if (!data.clusters) {
		return;
	}

	map<set<uint64_t>, CellConnectionItem*> newConnectionsByIds;
	for (auto const &cluster : *data.clusters) {
		for (auto const &cell : *cluster.cells) {
			if (!cell.connectingCells) {
				continue;
			}
			for (uint64_t connectingCellId : *cell.connectingCells) {
				auto &connectingCellD = visualDesc->getCellDescRef(connectingCellId);
				set<uint64_t> connectionId;
				connectionId.insert(cell.id);
				connectionId.insert(connectingCellId);
				if (newConnectionsByIds.find(connectionId) != newConnectionsByIds.end()) {
					continue;
				}
				//update may lead to exception in rare cases (Qt bug?)
/*
				auto connectionIt = _connectionsByIds.find(connectionId);
				if (connectionIt != _connectionsByIds.end()) {
					CellConnectionItem* connection = connectionIt->second;
					connection->update(cell, connectingCellD);
					newConnectionsByIds[connectionId] = connection;
					_connectionsByIds.erase(connectionIt);
				}
				else {
*/
					CellConnectionItem* newConnection = new CellConnectionItem(_config, cell, connectingCellD);
					_scene->addItem(newConnection);
					newConnectionsByIds[connectionId] = newConnection;
/*
				}
*/
			}
		}
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_connectionsByIds = newConnectionsByIds;
}

void ItemManager::update(DataController* visualDesc)
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

void ItemManager::toggleCellInfo(bool showInfo)
{
	_config->setShowCellInfo(showInfo);
}
