#ifndef ALIENENERGYGRAPHICSITEM_H
#define ALIENENERGYGRAPHICSITEM_H

#include <QGraphicsItem>

class AlienEnergy;
class AlienEnergyGraphicsItem : public QGraphicsItem
{
public:
    enum FocusState {
        NO_FOCUS,
        FOCUS
    };

    enum {
        Type = UserType + 2
    };

    AlienEnergyGraphicsItem (QGraphicsItem* parent = 0);
    AlienEnergyGraphicsItem (AlienEnergy* e, qreal x, qreal y, QGraphicsItem* parent = 0);
    ~AlienEnergyGraphicsItem ();

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    void setFocusState (FocusState focusState);
    AlienEnergy* getEnergyParticle ();

private:
    AlienEnergy* _e;
    FocusState _focusState;
};

#endif // ALIENENERGYGRAPHICSITEM_H
