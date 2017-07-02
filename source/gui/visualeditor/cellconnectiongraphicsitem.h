#pragma once

#include <QGraphicsItem>

#include "Model/Definitions.h"
#include "Gui/Definitions.h"

class CellConnectionGraphicsItem : public QGraphicsItem
{
public:
    enum ConnectionState {
        NO_DIR_CONNECTION,
        A_TO_B_CONNECTION,
        B_TO_A_CONNECTION
    };

	CellConnectionGraphicsItem(GraphicsItemConfig *config, CellDescription const &cell1, CellDescription const &cell2, QGraphicsItem *parent = nullptr);
	~CellConnectionGraphicsItem() = default;

	void update(CellDescription const &cell1, CellDescription const &cell2);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    void setConnectionState (ConnectionState connectionState);

private:
	GraphicsItemConfig* _config;

    qreal _dx = 0.0;
    qreal _dy = 0.0;
	ConnectionState _connectionState = ConnectionState::NO_DIR_CONNECTION;
};

