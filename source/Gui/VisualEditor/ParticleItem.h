#pragma once

#include "AbstractItem.h"

class ParticleItem
	: public AbstractItem
{
public:
    enum FocusState {
        NO_FOCUS,
        FOCUS
    };

    enum {
        Type = UserType + 2
    };

	ParticleItem(ItemConfig* config, EnergyParticleDescription const &desc, QGraphicsItem *parent = nullptr);
	virtual ~ParticleItem() = default;

	void update(EnergyParticleDescription const& desc);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    void setFocusState (FocusState focusState);

private:
	FocusState _focusState = FocusState::NO_FOCUS;
};

