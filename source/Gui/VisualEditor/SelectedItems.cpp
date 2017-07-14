#include <QGraphicsItem>

#include "SelectedItems.h"
#include "cellgraphicsitem.h"
#include "particlegraphicsitem.h"

namespace
{
	list<CellGraphicsItem*> getClusterOfCell(CellGraphicsItem *cell, map<uint64_t, uint64_t> const &clusterIdsByCellIds
		, map<uint64_t, CellGraphicsItem*> const &cellsByIds)
	{
		uint64_t clusterId = clusterIdsByCellIds.at(cell->getId());
		
		list<CellGraphicsItem*> result;
		for (auto clusterIdByCellId : clusterIdsByCellIds) {
			if (clusterIdByCellId.second == clusterId) {
				uint64_t cellId = clusterIdByCellId.first;
				result.push_back(cellsByIds.at(cellId));
			}
		}
		return result;
	}
}

void SelectedItems::set(list<QGraphicsItem*> const &items, map<uint64_t, uint64_t> const &clusterIdsByCellIds
	, map<uint64_t, CellGraphicsItem*> const &cellsByIds)
{
	unhighlightItems();

	_cells.clear();
	_clusters.clear();
	_particles.clear();

	for (auto item : items) {
		if (auto cellItem = qgraphicsitem_cast<CellGraphicsItem*>(item)) {
			_cells.push_back(cellItem);
			auto clusterItems = getClusterOfCell(cellItem, clusterIdsByCellIds, cellsByIds);
			_clusters.insert(_clusters.end(), clusterItems.begin(), clusterItems.end());
		}
		if (auto particleItem = qgraphicsitem_cast<ParticleGraphicsItem*>(item)) {
			_particles.push_back(particleItem);
		}
	}

	highlightItems();
}

void SelectedItems::unhighlightItems()
{
	for (auto cellItem : _clusters) {
		cellItem->setFocusState(CellGraphicsItem::NO_FOCUS);
	}
	for (auto particleItem : _particles) {
		particleItem->setFocusState(ParticleGraphicsItem::NO_FOCUS);
	}
}

void SelectedItems::highlightItems()
{
	for (auto cellItem : _clusters) {
		cellItem->setFocusState(CellGraphicsItem::FOCUS_CLUSTER);
	}
	for (auto cellItem : _cells) {
		cellItem->setFocusState(CellGraphicsItem::FOCUS_CELL);
	}
	for (auto particleItem : _particles) {
		particleItem->setFocusState(ParticleGraphicsItem::FOCUS);
	}
}
