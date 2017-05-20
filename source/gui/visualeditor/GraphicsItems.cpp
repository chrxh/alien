#include <QGraphicsScene>

#include "Gui/settings.h"
#include "GraphicsItems.h"
#include "cellgraphicsitem.h"

void GraphicsItems::init(QGraphicsScene * scene)
{
	_scene = scene;
}

void GraphicsItems::addCellItem(CellDescription const &desc)
{
	CellGraphicsItem* cellItem = new CellGraphicsItem(&_config, desc);	_scene->addItem(cellItem);
}
