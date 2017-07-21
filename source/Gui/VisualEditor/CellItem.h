#pragma once

#include "AbstractItem.h"

class CellItem
	: public AbstractItem
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

    CellItem (ItemConfig *config, CellDescription const &desc, QGraphicsItem *parent = nullptr);
    virtual ~CellItem () = default;

	virtual void update(CellDescription const &desc);

	virtual QRectF boundingRect () const;
	virtual void paint (QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);
	virtual int type() const;

	virtual uint64_t getId() const;
	virtual std::vector<uint64_t> getConnectedIds() const;
	virtual void setConnectable(bool connectable);
	virtual FocusState getFocusState ();
	virtual void setFocusState (FocusState focusState);
	virtual void setNumToken (int numToken);
	virtual void setColor (quint8 color);
	virtual void setDisplayString (QString value);
	virtual void setBranchNumber (int value);

private:
	ItemConfig *_config = nullptr;
    bool _connectable = false;
    FocusState _focusState = FocusState::NO_FOCUS;
    int _numToken = 0;
    quint8 _color = 0;
	QString _displayString;
	int _branchNumber = 0;
	uint64_t _id = 0;
	std::vector<uint64_t> _connectedIds;
};
