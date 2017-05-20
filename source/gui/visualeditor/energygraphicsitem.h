#ifndef ENERGYGRAPHICSITEM_H
#define ENERGYGRAPHICSITEM_H

#include <QGraphicsItem>

#include "Model/Definitions.h"

class EnergyGraphicsItem
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

    EnergyGraphicsItem(EnergyParticleDescription const &desc, QGraphicsItem *parent = nullptr);
	~EnergyGraphicsItem() = default;

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    void setFocusState (FocusState focusState);

private:
    FocusState _focusState;
};

#endif // ENERGYGRAPHICSITEM_H
