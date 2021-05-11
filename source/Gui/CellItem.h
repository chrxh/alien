#pragma once

#include "EngineInterface/ChangeDescriptions.h"

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
	virtual list<uint64_t> getConnectedIds() const;
	virtual FocusState getFocusState ();
	virtual void setFocusState (FocusState focusState);
	virtual void setDisplayString (QString value);

private:
    void adaptColorForFocus(QColor& color) const;
    void adaptColorForBoundary(QColor& color) const;
	int getBranchNumber() const;
	int getNumToken() const;
	bool isConnectable() const;
	uint8_t getColorCode() const;

	ItemConfig *_config = nullptr;
    FocusState _focusState = FocusState::NO_FOCUS;
	QString _displayString;
	CellDescription _desc;
};
