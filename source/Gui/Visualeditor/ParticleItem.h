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

	ParticleItem(ItemConfig* config, ParticleDescription const &desc, QGraphicsItem *parent = nullptr);
	virtual ~ParticleItem() = default;

	virtual void update(ParticleDescription const& desc);

	virtual uint64_t getId() const;

	virtual QRectF boundingRect () const;
	virtual void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
	virtual int type() const;

	virtual void setFocusState (FocusState focusState);

private:
	FocusState _focusState = FocusState::NO_FOCUS;
	uint64_t _id;
};

