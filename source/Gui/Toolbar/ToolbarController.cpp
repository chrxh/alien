#include "ToolbarController.h"

#include "ToolbarView.h"

ToolbarController::ToolbarController(IntVector2D const& upperLeftPosition, QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_view->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _view->width(), _view->height());
}

ToolbarController::~ToolbarController() {
	
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}
