#pragma once

#include <QGraphicsItem>

class MarkerItem 
	: public QGraphicsItem
{
public:
    MarkerItem (QPointF const &upperLeft, QPointF const &lowerRight, QGraphicsItem* parent = 0);
    virtual ~MarkerItem () = default;

	virtual void update(QPointF upperLeft, QPointF lowerRight);

    virtual void setLowerRight(QPointF lowerRight);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

private:
    qreal _x1;
    qreal _y1;
    qreal _dx;
    qreal _dy;
};
