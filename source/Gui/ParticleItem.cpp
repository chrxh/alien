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
    QColor color = Const::EnergyColor;

    int h, s, l;
    color.getHsl(&h, &s, &l);
    if (FocusState::FOCUS == _focusState) {
        l = 190;
    }
    color.setHsl(h, s, l);
    painter->setBrush(QBrush(color));

    if( _focusState == NO_FOCUS ) {
        l = std::max(0, l - 60);
        color.setHsl(h, s, l);
        painter->setPen(QPen(QBrush(color), CoordinateSystem::modelToScene(0.05)));
        painter->drawEllipse(
            QPointF(0.0, 0.0), CoordinateSystem::modelToScene(0.20), CoordinateSystem::modelToScene(0.20));
    }
    else {
        painter->setPen(QPen(QBrush(Const::EnergyPenFocusColor), CoordinateSystem::modelToScene(0.05)));
        painter->drawEllipse(QPointF(0.0, 0.0), CoordinateSystem::modelToScene(0.3), CoordinateSystem::modelToScene(0.3));
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



