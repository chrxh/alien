#include "ToolbarController.h"

#include "ToolbarView.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(IntVector2D const& upperLeftPosition, QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_view->setGeometry(upperLeftPosition.x, upperLeftPosition.y, _view->width(), _view->height());
	_context = new ToolbarContext(this);

	connect(_context, &ToolbarContext::activate, this, &ToolbarController::onActivate);
	connect(_context, &ToolbarContext::deactivate, this, &ToolbarController::onDeactivate);

	onDeactivate();
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}

void ToolbarController::onActivate()
{
	_view->setVisible(true);
}

void ToolbarController::onDeactivate()
{
	_view->setVisible(false);
}
