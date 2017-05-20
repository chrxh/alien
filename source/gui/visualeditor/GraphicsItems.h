#ifndef GRAPHICSITEMS_H
#define GRAPHICSITEMS_H

#include "gui/definitions.h"
#include "Model/Definitions.h"

#include "cellgraphicsitemconfig.h"

class GraphicsItems
	: public QObject
{
	Q_OBJECT
public:
	GraphicsItems(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GraphicsItems() = default;

	virtual void init(QGraphicsScene* scene);

	virtual void addCellItem(CellDescription const &desc);

private:
	QGraphicsScene* _scene = nullptr;

	CellGraphicsItemConfig _config;
};

#endif // GRAPHICSITEMS_H

