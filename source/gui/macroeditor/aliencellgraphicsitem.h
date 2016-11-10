#ifndef CELLGRAPHICSITEM_H
#define CELLGRAPHICSITEM_H

#include <QGraphicsItem>

class Cell;
class CellGraphicsItem : public QGraphicsItem
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

    CellGraphicsItem (QGraphicsItem* parent = 0);
    CellGraphicsItem (Cell* cell, qreal x, qreal y, bool connectable, int numToken, quint8 color, QGraphicsItem* parent = 0);
    ~CellGraphicsItem ();

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

    Cell* getCell ();
    void setConnectable (bool connectable);
    FocusState getFocusState ();
    void setFocusState (FocusState focusState);
    void setNumToken (int numToken);
    void setColor (quint8 color);

private:
    Cell* _cell;
    bool _connectable;
    FocusState _focusState;
    int _numToken;
    quint8 _color;
};

#endif // CELLGRAPHICSITEM_H
