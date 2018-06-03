#include <QPainter>

#include "gui/Settings.h"

#include "MarkerItem.h"
#include "CoordinateSystem.h"

MarkerItem::MarkerItem (QPointF const &upperLeft, QPointF const &lowerRight, QGraphicsItem* parent)
    : QGraphicsItem(parent)
{
	update(upperLeft, lowerRight);
}

void MarkerItem::update(QPointF upperLeft, QPointF lowerRight)
{
	upperLeft = CoordinateSystem::modelToScene(upperLeft);
	lowerRight = CoordinateSystem::modelToScene(lowerRight);
	_x1 = upperLeft.x();
	_y1 = upperLeft.y();
	_dx = lowerRight.x() - upperLeft.x();
	_dy = lowerRight.y() - upperLeft.y();
	QGraphicsItem::setPos(_x1, _y1);
}

void MarkerItem::setLowerRight(QPointF lowerRight)
{
	lowerRight = CoordinateSystem::modelToScene(lowerRight);
	_dx = lowerRight.x() - _x1;
    _dy = lowerRight.y() - _y1;
}

QRectF MarkerItem::boundingRect () const
{
    qreal minX = qMin(0.0, _dx);
    qreal minY = qMin(0.0, _dy);
    qreal maxX = qMax(0.0, _dx);
    qreal maxY = qMax(0.0, _dy);
    return QRectF(minX, minY, (maxX-minX), (maxY-minY));
}

void MarkerItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->setPen(QPen(QBrush(Const::MarkerColor), 0.00));
    painter->setBrush(QBrush(Const::MarkerColor));
    painter->drawRect(QRectF(0.0, 0.0, _dx, _dy));
}
