#include "ToolbarController.h"

#include "ToolbarView.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_context = new ToolbarContext(this);
}

void ToolbarController::init(IntVector2D const & upperLeftPosition)
{
	_view->init(upperLeftPosition);
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
