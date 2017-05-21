#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"

#include "GraphicsItemManager.h"
#include "cellgraphicsitem.h"
#include "ParticleGraphicsItem.h"

void GraphicsItemManager::init(QGraphicsScene * scene)
{
	_scene = scene;
}

void GraphicsItemManager::activate(IntVector2D size)
{
	_scene->clear();
	_scene->setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);
}

template<typename ItemType, typename DescriptionType>
void GraphicsItemManager::updateItems(vector<TrackerElement<DescriptionType>> const &desc, unordered_map<uint64_t, ItemType*>& itemsByIds
	, unordered_map<uint64_t, ItemType*>& newItemsByIds)
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

void GraphicsItemManager::update(DataDescription const &desc)
{
	unordered_map<uint64_t, CellGraphicsItem*> newCellsByIds;
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		updateItems(cluster.cells, _cellsByIds, newCellsByIds);
	}
	for (auto const& cellById : _cellsByIds) {
		delete cellById.second;
	}
	_cellsByIds = newCellsByIds;


	unordered_map<uint64_t, ParticleGraphicsItem*> newParticlesByIds;
	updateItems(desc.particles, _particlesByIds, newParticlesByIds);
	for (auto const& particleById : _particlesByIds) {
		delete particleById.second;
	}

	_particlesByIds = newParticlesByIds;
}
