#ifndef MARKERGRAPHICSITEM_H
#define MARKERGRAPHICSITEM_H

#include <QGraphicsItem>

class MarkerItem 
	: public QGraphicsItem
{
public:
    MarkerItem (qreal x1, qreal y1, qreal x2, qreal y2, QGraphicsItem* parent = 0);
    virtual ~MarkerItem () = default;

    void setEndPos(qreal x, qreal y);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

private:
    qreal _x1;
    qreal _y1;
    qreal _dx;
    qreal _dy;
};

#endif // MARKERGRAPHICSITEM_H
