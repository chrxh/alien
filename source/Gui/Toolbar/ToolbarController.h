#pragma once
#include <QWidget>

#include "Gui/Definitions.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(QWidget * parent = nullptr);
	virtual ~ToolbarController() = default;

	void init(IntVector2D const& upperLeftPosition);

	ToolbarContext* getContext() const;

private:
	Q_SLOT void onShow(bool visible);

	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
};
