#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"
#include "Gui/visualeditor/ViewportInterface.h"

#include "GraphicsItemManager.h"
#include "cellgraphicsitem.h"
#include "ParticleGraphicsItem.h"
#include "cellconnectiongraphicsitem.h"

void GraphicsItemManager::init(QGraphicsScene * scene, ViewportInterface* viewport)
{
	_scene = scene;
	_viewport = viewport;
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
			ItemType* newItem = new ItemType(&_config, descElement);
			_scene->addItem(newItem);
			newItemsByIds[descElement.id] = newItem;
		}
	}
}

void GraphicsItemManager::updateConnections(vector<TrackerElement<pair<uint64_t, uint64_t>>> const &desc
	, map<uint64_t, CellDescription> const &cellsByIds
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& connectionsByIds
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& newConnectionsByIds)
{
	for (auto const &connectionT : desc) {
		pair<uint64_t, uint64_t> const &connectionD = connectionT.getValue();
		auto firstCellIt = cellsByIds.find(connectionD.first);
		auto secondCellIt = cellsByIds.find(connectionD.second);
		if (firstCellIt == cellsByIds.end() || secondCellIt == cellsByIds.end()) {
			continue;
		}
		set<uint64_t> key;
		key.insert(connectionD.first);
		key.insert(connectionD.second);
		auto connectionIt = connectionsByIds.find(key);
		if (connectionIt != connectionsByIds.end()) {
			CellConnectionGraphicsItem* item = connectionIt->second;
			item->update(firstCellIt->second, secondCellIt->second);
			newConnectionsByIds[key] = item;
			connectionsByIds.erase(connectionIt);
		}
		else {
			CellConnectionGraphicsItem* newConnection = new CellConnectionGraphicsItem(&_config, firstCellIt->second, secondCellIt->second);
			_scene->addItem(newConnection);
			newConnectionsByIds[key] = newConnection;
		}
	}
}

/*
template<>
void GraphicsItemManager::updateEntities(vector<TrackerElement<pair<uint64_t, uint64_t>>> const &desc
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& itemsByIds
	, map<set<uint64_t>, CellConnectionGraphicsItem*>& newItemsByIds)
{
	for (auto const &descElementT : desc) {

		auto const &descElement = descElementT.getValue();
		set<uint64_t> key;
		key.insert(descElement.first);
		key.insert(descElement.second);
		auto it = itemsByIds.find(key);

		if (it != itemsByIds.end()) {
			auto item = it->second;
//			item->update(descElement);
			newItemsByIds[key] = item;
			itemsByIds.erase(it);
		}
		else {
			CellGraphicsItem* item1 = 
			CellConnectionGraphicsItem* newItem = new CellConnectionGraphicsItem(&_config, );
			_scene->addItem(newItem);
			newItemsByIds[key] = newItem;
		}
	}
}
*/

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
}

void GraphicsItemManager::update(DataDescription const &desc)
{
	_viewport->setModeToNoUpdate();

	map<uint64_t, CellDescription> cellDescByIds;
	getCellsByIds(desc, cellDescByIds);

	map<uint64_t, CellGraphicsItem*> newCellsByIds;
	map<set<uint64_t>, CellConnectionGraphicsItem*> newConnectionsByIds;
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		updateEntities(cluster.cells, _cellsByIds, newCellsByIds);
		updateConnections(cluster.cellConnections, cellDescByIds, _connectionsByIds, newConnectionsByIds);
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

/*
	map<set<uint64_t>, CellConnectionGraphicsItem*> newConnectionsByIds;
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		updateItems(cluster.cellConnections, _cellsByIds, newCellsByIds);
	}
	for (auto const& connectionById : _connectionsByIds) {
		delete connectionById.second;
	}
	_connectionsByIds = newConnectionsByIds;
*/

	_viewport->setModeToUpdate();
}
