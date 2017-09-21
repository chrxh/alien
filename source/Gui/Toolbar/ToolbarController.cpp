#include "ToolbarController.h"

#include "ToolbarView.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(IntVector2D const& upperLeftPosition, QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_view->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _view->width(), _view->height());
	_context = new ToolbarContext(this);

	connect(_context, &ToolbarContext::show, this, &ToolbarController::onShow);

	onShow(false);
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}

void ToolbarController::onShow(bool visible)
{
	_view->setVisible(visible);
}
