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
void GraphicsItemManager::updateItems(vector<TrackerElement<DescriptionType>> const &desc
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

/*
template<>
void GraphicsItemManager::updateItems(vector<TrackerElement<pair<uint64_t, uint64_t>>> const &desc
	, map<unordered_set<uint64_t>, CellConnectionGraphicsItem*>& itemsByIds
	, map<unordered_set<uint64_t>, CellConnectionGraphicsItem*>& newItemsByIds)
{
	for (auto const &descElementT : desc) {
		auto const &descElement = descElementT.getValue();
		unordered_set<uint64_t> key;
		key.insert(descElement.first);
		key.insert(descElement.second);
		auto it = itemsByIds.find(key);
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
*/

void GraphicsItemManager::update(DataDescription const &desc)
{
	_viewport->setModeToNoUpdate();

	map<uint64_t, CellGraphicsItem*> newCellsByIds;
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		updateItems(cluster.cells, _cellsByIds, newCellsByIds);
	}
	for (auto const& cellById : _cellsByIds) {
		delete cellById.second;
	}
	_cellsByIds = newCellsByIds;

	map<uint64_t, ParticleGraphicsItem*> newParticlesByIds;
	updateItems(desc.particles, _particlesByIds, newParticlesByIds);
	for (auto const& particleById : _particlesByIds) {
		delete particleById.second;
	}
	_particlesByIds = newParticlesByIds;

/*
	map<unordered_set<uint64_t>, CellConnectionGraphicsItem*> newConnectionsByIds;
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
