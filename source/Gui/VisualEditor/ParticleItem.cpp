#include <QPainter>

#include "Model/Entities/Descriptions.h"
#include "gui/Settings.h"

#include "ParticleItem.h"

ParticleItem::ParticleItem (ItemConfig* config, EnergyParticleDescription const &desc, QGraphicsItem *parent /*= nullptr*/)
    : AbstractItem(parent)
{
	update(desc);
}

void ParticleItem::update(EnergyParticleDescription const & desc)
{
	_id = desc.id;
	auto pos = desc.pos.getValue();
	QGraphicsItem::setPos(QPointF(pos.x()*GRAPHICS_ITEM_SIZE, pos.y()*GRAPHICS_ITEM_SIZE));
}

uint64_t ParticleItem::getId() const
{
	return _id;
}

QRectF ParticleItem::boundingRect () const
{
    return QRectF(-0.4*GRAPHICS_ITEM_SIZE, -0.4*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE, 0.8*GRAPHICS_ITEM_SIZE);
}

void ParticleItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
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

int ParticleItem::type() const
{
    return Type;
}

void ParticleItem::setFocusState (FocusState focusState)
{
    _focusState = focusState;
}



