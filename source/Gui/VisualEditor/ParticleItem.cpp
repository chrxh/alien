#include <QPainter>

#include "Model/Entities/ChangeDescriptions.h"
#include "gui/Settings.h"

#include "ParticleItem.h"
#include "CoordinateSystem.h"

ParticleItem::ParticleItem (ItemConfig* config, ParticleDescription const &desc, QGraphicsItem *parent /*= nullptr*/)
    : AbstractItem(parent)
{
	update(desc);
}

void ParticleItem::update(ParticleDescription const & desc)
{
	_id = desc.id;
	auto pos = CoordinateSystem::modelToScene(*desc.pos);
	QGraphicsItem::setPos(QPointF(pos.x(), pos.y()));
}

uint64_t ParticleItem::getId() const
{
	return _id;
}

QRectF ParticleItem::boundingRect () const
{
    return QRectF(CoordinateSystem::modelToScene(-0.5), CoordinateSystem::modelToScene(-0.5)
		, CoordinateSystem::modelToScene(1.0), CoordinateSystem::modelToScene(1.0));
}

void ParticleItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if( _focusState == NO_FOCUS ) {
        painter->setPen(QPen(QBrush(QColor(0,0,0,0)), CoordinateSystem::modelToScene(0.03)));
        painter->setBrush(QBrush(ENERGY_COLOR));
        painter->drawEllipse(QPointF(0.0, 0.0), CoordinateSystem::modelToScene(0.20), CoordinateSystem::modelToScene(0.20));
    }
    else {
        painter->setPen(QPen(QBrush(ENERGY_PEN_FOCUS_COLOR), CoordinateSystem::modelToScene(0.03)));
        painter->setBrush(QBrush(ENERGY_FOCUS_COLOR));
        painter->drawEllipse(QPointF(0.0, 0.0), CoordinateSystem::modelToScene(0.4), CoordinateSystem::modelToScene(0.4));
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



