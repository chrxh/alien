#include "ToolbarController.h"

ToolbarController::ToolbarController(QWidget* parent) : QObject(parent) {
	
}

ToolbarController::~ToolbarController() {
	
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}
