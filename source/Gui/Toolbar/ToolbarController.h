#pragma once
#include <QWidget>

#include "Gui/Definitions.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(IntVector2D const& upperLeftPosition, QWidget * parent = nullptr);
	~ToolbarController() = default;

	ToolbarContext* getContext() const;

private:
	Q_SLOT void onActivate();
	Q_SLOT void onDeactivate();

	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
};
