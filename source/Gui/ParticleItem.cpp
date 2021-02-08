#include <QPainter>

#include "EngineInterface/ChangeDescriptions.h"
#include "Gui/Settings.h"

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
    return QRectF(CoordinateSystem::modelToScene(-0.4), CoordinateSystem::modelToScene(-0.4)
		, CoordinateSystem::modelToScene(0.8), CoordinateSystem::modelToScene(0.8));
}

void ParticleItem::paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    if( _focusState == NO_FOCUS ) {
        painter->setPen(QPen(QBrush(QColor(0,0,0,0)), CoordinateSystem::modelToScene(0.03)));
        painter->setBrush(QBrush(Const::EnergyColor));
        painter->drawEllipse(QPointF(0.0, 0.0), CoordinateSystem::modelToScene(0.20), CoordinateSystem::modelToScene(0.20));
    }
    else {
        painter->setPen(QPen(QBrush(Const::EnergyPenFocusColor), CoordinateSystem::modelToScene(0.03)));
        painter->setBrush(QBrush(Const::EnergyFocusColor));
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



