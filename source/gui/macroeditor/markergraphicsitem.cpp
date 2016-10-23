#include "markergraphicsitem.h"

#include "global/editorsettings.h"

#include <QPainter>

MarkerGraphicsItem::MarkerGraphicsItem (qreal x1, qreal y1, qreal x2, qreal y2, QGraphicsItem* parent)
    : QGraphicsItem(parent), _x1(x1), _y1(y1), _dx(x2-x1), _dy(y2-y1)
{
    QGraphicsItem::setPos(x1, y1);
}

MarkerGraphicsItem::~MarkerGraphicsItem ()
{

}

void MarkerGraphicsItem::setEndPos(qreal x, qreal y)
{
    _dx = x-_x1;
    _dy = y-_y1;
}

QRectF MarkerGraphicsItem::boundingRect () const
{
    qreal minX = qMin(0.0, _dx);
    qreal minY = qMin(0.0, _dy);
    qreal maxX = qMax(0.0, _dx);
    qreal maxY = qMax(0.0, _dy);
    return QRectF(minX, minY, (maxX-minX), (maxY-minY));
}

void MarkerGraphicsItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    painter->setPen(QPen(QBrush(MARKER_COLOR), 0.00));
    painter->setBrush(QBrush(MARKER_COLOR));
    painter->drawRect(QRectF(0.0, 0.0, _dx, _dy));
}
