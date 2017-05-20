#ifndef GRAPHICSITEMMANAGER_H
#define GRAPHICSITEMMANAGER_H

#include "gui/definitions.h"
#include "Model/Definitions.h"

#include "cellgraphicsitemconfig.h"

class GraphicsItemManager
	: public QObject
{
	Q_OBJECT
public:
	GraphicsItemManager(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~GraphicsItemManager() = default;

	virtual void init(QGraphicsScene* scene);

	virtual void setActive(IntVector2D size);
	virtual void buildItems(DataDescription const &desc);

private:
	QGraphicsScene* _scene = nullptr;

	CellGraphicsItemConfig _config;
};

#endif // GRAPHICSITEMMANAGER_H

