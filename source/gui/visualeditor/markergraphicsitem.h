#ifndef MARKERGRAPHICSITEM_H
#define MARKERGRAPHICSITEM_H

#include <QGraphicsItem>

class MarkerGraphicsItem : public QGraphicsItem
{
public:
    MarkerGraphicsItem (qreal x1, qreal y1, qreal x2, qreal y2, QGraphicsItem* parent = 0);
    ~MarkerGraphicsItem ();

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
