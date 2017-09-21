#include "CoordinateSystem.h"

const qreal GRAPHICS_ITEM_SIZE = 10.0;

QVector2D CoordinateSystem::sceneToModel(QVector2D const & vec)
{
	return QVector2D(vec.x() / GRAPHICS_ITEM_SIZE, vec.y() / GRAPHICS_ITEM_SIZE);
}

QPointF CoordinateSystem::sceneToModel(QPointF const & p)
{
	return QPointF(p.x() / GRAPHICS_ITEM_SIZE, p.y() / GRAPHICS_ITEM_SIZE);
}

double CoordinateSystem::sceneToModel(double len)
{
	return len / GRAPHICS_ITEM_SIZE;
}

QVector2D CoordinateSystem::modelToScene(QVector2D const & vec)
{
	return QVector2D(vec.x() * GRAPHICS_ITEM_SIZE, vec.y() * GRAPHICS_ITEM_SIZE);
}

QPointF CoordinateSystem::modelToScene(QPointF const & p)
{
	return QPointF(p.x() * GRAPHICS_ITEM_SIZE, p.y() * GRAPHICS_ITEM_SIZE);
}


double CoordinateSystem::modelToScene(double len)
{
	return len * GRAPHICS_ITEM_SIZE;
}
