#include <QPainter>

#include "Model/Entities/Descriptions.h"
#include "gui/Settings.h"
#include "gui/Settings.h"

#include "energygraphicsitem.h"

EnergyGraphicsItem::EnergyGraphicsItem (EnergyParticleDescription const &desc, QGraphicsItem *parent /*= nullptr*/)
    : QGraphicsItem(parent), _focusState(NO_FOCUS)
{
	auto pos = desc.pos.getValue()*GRAPHICS_ITEM_SIZE;
    QGraphicsItem::setPos(pos.x(), pos.y());
}

QRectF EnergyGraphicsItem::boundingRect () const
{
    return QRectF(-0.4*GRAPHICS_ITEM_SIZE, -0.4*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE);
}

void EnergyGraphicsItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if( _focusState == NO_FOCUS ) {
        painter->setPen(QPen(QBrush(QColor(0,0,0,0)), 0.03));
        painter->setBrush(QBrush(ENERGY_COLOR));
        painter->drawEllipse(QPointF(0.0, 0.0), 0.20*GRAPHICS_ITEM_SIZE, 0.20*GRAPHICS_ITEM_SIZE);
    }
    else {
        painter->setPen(QPen(QBrush(ENERGY_PEN_FOCUS_COLOR), 0.03));
        painter->setBrush(QBrush(ENERGY_FOCUS_COLOR));
        painter->drawEllipse(QPointF(0.0, 0.0), 0.4*GRAPHICS_ITEM_SIZE, 0.4*GRAPHICS_ITEM_SIZE);
    }
}

int EnergyGraphicsItem::type() const
{
    return Type;
}

void EnergyGraphicsItem::setFocusState (FocusState focusState)
{
    _focusState = focusState;
}



