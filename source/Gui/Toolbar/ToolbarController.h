#pragma once
#include <QWidget>

#include "Gui/Definitions.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(IntVector2D const& upperLeftPosition, QWidget * parent = nullptr);
	~ToolbarController();

	ToolbarContext* getContext() const;

private:
	ToolbarContext* _context = nullptr;
	ToolbarView* _view = nullptr;
};
