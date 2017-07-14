#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "GraphicsItemManager.h"
#include "cellgraphicsitem.h"
#include "ParticleGraphicsItem.h"
#include "cellconnectiongraphicsitem.h"

void GraphicsItemManager::init(QGraphicsScene * scene, ViewportInterface* viewport, SimulationParameters* parameters)
{
	auto config = new GraphicsItemConfig();

	_scene = scene;
	_viewport = viewport;
	_parameters = parameters;
	SET_CHILD(_config, config);

	_config->init(parameters);
}

void GraphicsItemManager::activate(IntVector2D size)
{
	_scene->clear();
	_scene->setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);
	_cellsByIds.clear();
	_particlesByIds.clear();
}

template<typename IdType, typename ItemType, typename DescriptionType>
void GraphicsItemManager::updateEntities(vector<TrackerElement<DescriptionType>> const &desc
	, map<IdType, ItemType*>& itemsByIds, map<IdType, ItemType*>& newItemsByIds)
{
	for (auto const &descElementT : desc) {
		auto const &descElement = descElementT.getValue();
		auto it = itemsByIds.find(descElement.id);
		if (it != itemsByIds.end()) {
			auto item = it->second;
			item->update(descElement);
			newItemsByIds[descElement.id] = item;
			itemsByIds.erase(it);
		}
		else {
			ItemType* newItem = new ItemType(_config, descElement);
			_scene->addItem(newItem);
			newItemsByIds[descElement.id] = newItem;
		}
	}
}

void GraphicsItemManager::updateConnections(vector<TrackerElement<CellDescription>> const &desc
	, map<uint64_t, CellDescription> const &cellsByIds
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& connectionsByIds
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& newConnectionsByIds)
{
	for (auto const &cellT : desc) {
		auto const &cellD = cellT.getValue();
		if (!cellD.connectingCells.isInitialized()) {
			continue;
		}
		for (uint64_t connectingCellId : cellD.connectingCells.getValue()) {
			auto cellIt = cellsByIds.find(connectingCellId);
			if (cellIt == cellsByIds.end()) {
				continue;
			}
			set<uint64_t> id;
			id.insert(cellD.id);
			id.insert(connectingCellId);
			if (newConnectionsByIds.find(id) != newConnectionsByIds.end()) {
				continue;
			}
			auto connectionIt = connectionsByIds.find(id);
			if (connectionIt != connectionsByIds.end()) {
				CellConnectionGraphicsItem* connection = connectionIt->second;
				connection->update(cellD, cellIt->second);
				newConnectionsByIds[id] = connection;
				connectionsByIds.erase(connectionIt);
			}
			else {
				CellConnectionGraphicsItem* newConnection = new CellConnectionGraphicsItem(_config, cellD, cellIt->second);
				_scene->addItem(newConnection);
				newConnectionsByIds[id] = newConnection;
			}
		}
	}
}

namespace
{
	void getCellsByIds(DataDescription const &desc, map<uint64_t, CellDescription> &result)
	{
		for (auto const &clusterT : desc.clusters) {
			auto const &clusterD = clusterT.getValue();
			for (auto const &cellT : clusterD.cells) {
				auto const &cellD = cellT.getValue();
				result[cellD.id] = cellD;
			}
		}
	}

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

void GraphicsItemManager::update(DataDescription const &desc)
{
	_viewport->setModeToNoUpdate();

	map<uint64_t, CellDescription> cellDescByIds;
	getCellsByIds(desc, cellDescByIds);
	getClusterIdsByCellIds(desc, _clusterIdsByCellIds);

	map<uint64_t, CellGraphicsItem*> newCellsByIds;
	map<set<uint64_t>, CellConnectionGraphicsItem*> newConnectionsByIds;
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		updateEntities(cluster.cells, _cellsByIds, newCellsByIds);
		updateConnections(cluster.cells, cellDescByIds, _connectionsByIds, newConnectionsByIds);
	}
	for (auto const& cellById : _cellsByIds) {
		delete cellById.second;
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_cellsByIds = newCellsByIds;
	_connectionsByIds = newConnectionsByIds;

	map<uint64_t, ParticleGraphicsItem*> newParticlesByIds;
	updateEntities(desc.particles, _particlesByIds, newParticlesByIds);
	for (auto const& particleById : _particlesByIds) {
		delete particleById.second;
	}
	_particlesByIds = newParticlesByIds;

	_viewport->setModeToUpdate();
}

void GraphicsItemManager::setSelection(list<QGraphicsItem*> const &items)
{
	_selectedItems.set(items, _clusterIdsByCellIds, _cellsByIds);
}
