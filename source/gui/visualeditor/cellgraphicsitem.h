#ifndef CELLGRAPHICSITEM_H
#define CELLGRAPHICSITEM_H

#include <QGraphicsItem>

#include "Model/Definitions.h"
#include "gui/Definitions.h"

class CellGraphicsItem
	: public QGraphicsItem
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

    CellGraphicsItem (GraphicsItemConfig* config, CellDescription const& desc, QGraphicsItem* parent = nullptr);
    ~CellGraphicsItem () = default;

	void update(CellDescription const& desc);

    QRectF boundingRect () const;
    void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
    int type() const;

	uint64_t getId() const;
    void setConnectable (bool connectable);
    FocusState getFocusState ();
    void setFocusState (FocusState focusState);
    void setNumToken (int numToken);
    void setColor (quint8 color);
	void setDisplayString (QString value);
	void setBranchNumber (int value);

private:
	GraphicsItemConfig* _config = nullptr;
    bool _connectable = false;
    FocusState _focusState = FocusState::NO_FOCUS;
    int _numToken = 0;
    quint8 _color = 0;
	QString _displayString;
	int _branchNumber = 0;
	uint64_t _id = 0;
};

#endif // CELLGRAPHICSITEM_H
