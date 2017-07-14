#pragma once

#include "Gui/Definitions.h"

class SelectedItems
{
public:
	virtual void set(list<QGraphicsItem*> const &items, map<uint64_t, uint64_t> const &clusterIdsByCellIds
		, map<uint64_t, CellGraphicsItem*> const &cellsByIds);

private:
	void unhighlightItems();
	void highlightItems();

	list<CellGraphicsItem*> _cells;
	list<CellGraphicsItem*> _clusters;
	list<ParticleGraphicsItem*> _particles;
};