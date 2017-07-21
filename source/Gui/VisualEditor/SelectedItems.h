#pragma once

#include "Gui/Definitions.h"

class SelectedItems
	: public QObject
{
	Q_OBJECT
public:
	SelectedItems(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~SelectedItems() = default;

	virtual void set(list<QGraphicsItem*> const &items, map<uint64_t, uint64_t> const &clusterIdsByCellIds
		, map<uint64_t, CellItem*> const &cellsByIds);
	virtual void move(QVector2D const &delta);

private:
	void unhighlightItems();
	void highlightItems();

	list<CellItem*> _cells;
	list<CellItem*> _clusters;
	list<ParticleItem*> _particles;
};