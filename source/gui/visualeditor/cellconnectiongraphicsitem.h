#ifndef CELLCONNECTIONGRAPHICSITEM_H
#define CELLCONNECTIONGRAPHICSITEM_H

#include <QGraphicsItem>

#include "Model/Definitions.h"

class CellConnectionGraphicsItem : public QGraphicsItem
{
public:
    enum ConnectionState {
        NO_DIR_CONNECTION,
        A_TO_B_CONNECTION,
        B_TO_A_CONNECTION
    };

	CellConnectionGraphicsItem(CellDescription const &cell1, CellDescription const &cell2, ConnectionState s, QGraphicsItem* parent = nullptr);
	CellConnectionGraphicsItem(qreal x1, qreal y1, qreal x2, qreal y2, ConnectionState s, QGraphicsItem* parent = nullptr);
	~CellConnectionGraphicsItem() = default;

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void setConnectionState (ConnectionState connectionState);

private:
    qreal _dx = 0.0;
    qreal _dy = 0.0;
	ConnectionState _connectionState = ConnectionState::NO_DIR_CONNECTION;
};

#endif // CELLCONNECTIONGRAPHICSITEM_H
