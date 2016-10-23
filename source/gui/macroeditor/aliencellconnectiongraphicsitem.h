#ifndef ALIENCELLCONNECTIONGRAPHICSITEM_H
#define ALIENCELLCONNECTIONGRAPHICSITEM_H

#include <QGraphicsItem>

class AlienCellConnectionGraphicsItem : public QGraphicsItem
{
public:
    enum ConnectionState {
        NO_DIR_CONNECTION,
        A_TO_B_CONNECTION,
        B_TO_A_CONNECTION
    };

    AlienCellConnectionGraphicsItem (qreal x1, qreal y1, qreal x2, qreal y2, ConnectionState s, QGraphicsItem* parent = 0);
    ~AlienCellConnectionGraphicsItem ();

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void setConnectionState (ConnectionState connectionState);

private:
    qreal _dx;
    qreal _dy;
    ConnectionState _connectionState;
};

#endif // ALIENCELLCONNECTIONGRAPHICSITEM_H
