#ifndef CELLGRAPHICSITEM_H
#define CELLGRAPHICSITEM_H

#include <QGraphicsItem>
#include "gui/definitions.h"

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

    CellGraphicsItem (CellGraphicsItemConfig* config, Cell* cell, qreal x, qreal y, bool connectable, int numToken, quint8 color
		, QString displayString, int branchNumber, QGraphicsItem* parent = 0);
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
	void setDisplayString (QString value);
	void setBranchNumber (int value);

private:
    Cell* _cell;
	CellGraphicsItemConfig* _config = nullptr;
    bool _connectable = false;
    FocusState _focusState;
    int _numToken;
    quint8 _color;
	QString _displayString;
	int _branchNumber;
};

#endif // CELLGRAPHICSITEM_H
