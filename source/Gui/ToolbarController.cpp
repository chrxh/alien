#include <QAction>

#include "ToolbarController.h"

#include "ModelInterface/SimulationContext.h"
#include "ModelInterface/SimulationParameters.h"

#include "Gui/DataRepository.h"
#include "Gui/Notifier.h"
#include "Gui/ActionHolder.h"
#include "ToolbarView.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_context = new ToolbarContext(this);
}

void ToolbarController::init(IntVector2D const & upperLeftPosition, Notifier* notifier, DataRepository* manipulator
	, SimulationContext const* context, ActionHolder* actions)
{
	_notifier = notifier;
	_repository = manipulator;
	_parameters = context->getSimulationParameters();
	_actions = actions;
	_view->init(upperLeftPosition, actions, this);

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

