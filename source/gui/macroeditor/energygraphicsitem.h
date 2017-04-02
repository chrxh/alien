#ifndef ENERGYGRAPHICSITEM_H
#define ENERGYGRAPHICSITEM_H

#include <QGraphicsItem>

class EnergyParticle;
class EnergyGraphicsItem : public QGraphicsItem
{
public:
    enum FocusState {
        NO_FOCUS,
        FOCUS
    };

    enum {
        Type = UserType + 2
    };

    EnergyGraphicsItem (QGraphicsItem* parent = 0);
    EnergyGraphicsItem (EnergyParticle* e, qreal x, qreal y, QGraphicsItem* parent = 0);
    ~EnergyGraphicsItem ();

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    void setFocusState (FocusState focusState);
    EnergyParticle* getEnergyParticle ();

private:
    EnergyParticle* _e;
    FocusState _focusState;
};

#endif // ENERGYGRAPHICSITEM_H
