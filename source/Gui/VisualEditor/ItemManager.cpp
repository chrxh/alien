#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "ItemManager.h"
#include "VisualDescription.h"
#include "CellItem.h"
#include "ParticleItem.h"
#include "CellConnectionItem.h"

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
	_scene->setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);
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
	auto cellDescsByIds = visualDesc->getCellDescsByIds();

	map<set<uint64_t>, CellConnectionItem*> newConnectionsByIds;
	for (auto const &clusterT : data.clusters) {
		auto const &cluster = clusterT.getValue();
		for (auto const &cellT : cluster.cells) {
			auto const &cellD = cellT.getValue();
			if (!cellD.connectingCells.isInitialized()) {
				continue;
			}
			for (uint64_t connectingCellId : cellD.connectingCells.getValue()) {
				auto cellIt = cellDescsByIds.find(connectingCellId);
				if (cellIt == cellDescsByIds.end()) {
					continue;
				}
				set<uint64_t> id;
				id.insert(cellD.id);
				id.insert(connectingCellId);
				if (newConnectionsByIds.find(id) != newConnectionsByIds.end()) {
					continue;
				}
				auto connectionIt = _connectionsByIds.find(id);
				if (connectionIt != _connectionsByIds.end()) {
					CellConnectionItem* connection = connectionIt->second;
					connection->update(cellD, cellIt->second);
					newConnectionsByIds[id] = connection;
					_connectionsByIds.erase(connectionIt);
				}
				else {
					CellConnectionItem* newConnection = new CellConnectionItem(_config, cellD, cellIt->second);
					_scene->addItem(newConnection);
					newConnectionsByIds[id] = newConnection;
				}
			}
		}
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_connectionsByIds = newConnectionsByIds;
}

namespace
{
	void getClusterIdsByCellIds(DataDescription const &desc, map<uint64_t, uint64_t> &result)
	{
		for (auto const &clusterT : desc.clusters) {
			auto const &cluster = clusterT.getValue();
			for (auto const &cellT : cluster.cells) {
				auto const &cell = cellT.getValue();
				result[cell.id] = cluster.id;
			}
		}
	}

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
