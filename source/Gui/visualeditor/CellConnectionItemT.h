#pragma once

#include "AbstractItemT.h"

class CellConnectionItem
	: public AbstractItem
{
public:
    enum ConnectionState {
        NO_DIR_CONNECTION,
        A_TO_B_CONNECTION,
        B_TO_A_CONNECTION
    };

	CellConnectionItem(ItemConfig *config, CellDescription const &cell1, CellDescription const &cell2, QGraphicsItem *parent = nullptr);
	virtual ~CellConnectionItem() = default;

	void update(CellDescription const &cell1, CellDescription const &cell2);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void setConnectionState (ConnectionState connectionState);

private:
	ItemConfig* _config;

    qreal _dx = 0.0;
    qreal _dy = 0.0;
	ConnectionState _connectionState = ConnectionState::NO_DIR_CONNECTION;
};

