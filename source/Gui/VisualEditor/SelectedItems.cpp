#include <QGraphicsItem>

#include "SelectedItems.h"
#include "CellItem.h"
#include "ParticleItem.h"

namespace
{
	list<CellItem*> getClusterOfCell(CellItem *cell, map<uint64_t, uint64_t> const &clusterIdsByCellIds
		, map<uint64_t, CellItem*> const &cellsByIds)
	{
		uint64_t clusterId = clusterIdsByCellIds.at(cell->getId());
		
		list<CellItem*> result;
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
	, map<uint64_t, CellItem*> const &cellsByIds)
{
	unhighlightItems();

	_cells.clear();
	_clusters.clear();
	_particles.clear();

	for (auto item : items) {
		if (auto cellItem = qgraphicsitem_cast<CellItem*>(item)) {
			_cells.push_back(cellItem);
			auto clusterItems = getClusterOfCell(cellItem, clusterIdsByCellIds, cellsByIds);
			_clusters.insert(_clusters.end(), clusterItems.begin(), clusterItems.end());
		}
		if (auto particleItem = qgraphicsitem_cast<ParticleItem*>(item)) {
			_particles.push_back(particleItem);
		}
	}

	highlightItems();
}

void SelectedItems::move(QVector2D const &delta)
{
	for (auto item : _cells) {
		item->move(delta);
	}
	for (auto item : _particles) {
		item->move(delta);
	}
}

void SelectedItems::unhighlightItems()
{
	for (auto cellItem : _clusters) {
		cellItem->setFocusState(CellItem::NO_FOCUS);
	}
	for (auto particleItem : _particles) {
		particleItem->setFocusState(ParticleItem::NO_FOCUS);
	}
}

void SelectedItems::highlightItems()
{
	for (auto cellItem : _clusters) {
		cellItem->setFocusState(CellItem::FOCUS_CLUSTER);
	}
	for (auto cellItem : _cells) {
		cellItem->setFocusState(CellItem::FOCUS_CELL);
	}
	for (auto particleItem : _particles) {
		particleItem->setFocusState(ParticleItem::FOCUS);
	}
}
