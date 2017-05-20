#include <QGraphicsScene>

#include "Model/Entities/Descriptions.h"
#include "Gui/settings.h"

#include "GraphicsItemManager.h"
#include "cellgraphicsitem.h"
#include "energygraphicsitem.h"

void GraphicsItemManager::init(QGraphicsScene * scene)
{
	_scene = scene;
}

void GraphicsItemManager::setActive(IntVector2D size)
{
	_scene->clear();
	_scene->setSceneRect(0, 0, size.x*GRAPHICS_ITEM_SIZE, size.y*GRAPHICS_ITEM_SIZE);
}

void GraphicsItemManager::buildItems(DataDescription const &desc)
{
	for (auto const &clusterT : desc.clusters) {
		auto const &cluster = clusterT.getValue();
		for (auto const &cellT : cluster.cells) {
			auto const &cell = cellT.getValue();
			CellGraphicsItem* cellItem = new CellGraphicsItem(&_config, cell);
			_scene->addItem(cellItem);
		}
	}
	for (auto const &particleT : desc.particles) {
		auto const &particle = particleT.getValue();
		EnergyGraphicsItem* eItem = new EnergyGraphicsItem(particle);
		_scene->addItem(eItem);
	}
}
