#ifndef ENERGYGRAPHICSITEM_H
#define ENERGYGRAPHICSITEM_H

#include <QGraphicsItem>

#include "Model/Definitions.h"
#include "Gui/Definitions.h"

class ParticleGraphicsItem
	: public QGraphicsItem
{
public:
    enum FocusState {
        NO_FOCUS,
        FOCUS
    };

    enum {
        Type = UserType + 2
    };

    ParticleGraphicsItem(GraphicsItemConfig* config, EnergyParticleDescription const &desc, QGraphicsItem *parent = nullptr);
	~ParticleGraphicsItem() = default;

	void update(EnergyParticleDescription const& desc);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    void setFocusState (FocusState focusState);

private:
    FocusState _focusState;
};

#endif // ENERGYGRAPHICSITEM_H
