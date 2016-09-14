#ifndef ALIENCELLGRAPHICSITEM_H
#define ALIENCELLGRAPHICSITEM_H

#include <QGraphicsItem>

class AlienCell;
class AlienCellGraphicsItem : public QGraphicsItem
{
public:
    enum FocusState {
        NO_FOCUS,
        FOCUS_CLUSTER,
        FOCUS_CELL
    };
    enum {
        Type = UserType + 1
    };

    AlienCellGraphicsItem (QGraphicsItem* parent = 0);
    AlienCellGraphicsItem (AlienCell* cell, qreal x, qreal y, bool connectable, int numToken, quint8 color, QGraphicsItem* parent = 0);
    ~AlienCellGraphicsItem ();

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    AlienCell* getCell ();
    void setConnectable (bool connectable);
    FocusState getFocusState ();
    void setFocusState (FocusState focusState);
    void setNumToken (int numToken);
    void setColor (quint8 color);

private:
    AlienCell* _cell;
    bool _connectable;
    FocusState _focusState;
    int _numToken;
    quint8 _color;
};

#endif // ALIENCELLGRAPHICSITEM_H
