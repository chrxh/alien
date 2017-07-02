#include <QPainter>

#include "Model/Entities/Descriptions.h"
#include "gui/Settings.h"

#include "ParticleGraphicsItem.h"

ParticleGraphicsItem::ParticleGraphicsItem (GraphicsItemConfig* config, EnergyParticleDescription const &desc, QGraphicsItem *parent /*= nullptr*/)
    : QGraphicsItem(parent)
{
	update(desc);
}

void ParticleGraphicsItem::update(EnergyParticleDescription const & desc)
{
	auto pos = desc.pos.getValue()*GRAPHICS_ITEM_SIZE;
	QGraphicsItem::setPos(pos.x(), pos.y());
}

QRectF ParticleGraphicsItem::boundingRect () const
{
    return QRectF(-0.4*GRAPHICS_ITEM_SIZE, -0.4*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE);
}

void ParticleGraphicsItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
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

int ParticleGraphicsItem::type() const
{
    return Type;
}

void ParticleGraphicsItem::setFocusState (FocusState focusState)
{
    _focusState = focusState;
}



