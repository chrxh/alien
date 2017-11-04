#include "ToolbarController.h"

#include "Gui/DataManipulator.h"

#include "ToolbarView.h"
#include "ToolbarModel.h"
#include "ToolbarContext.h"

ToolbarController::ToolbarController(QWidget* parent)
	: QObject(parent)
{
	_view = new ToolbarView(parent);
	_model = new ToolbarModel(this);
	_context = new ToolbarContext(this);
}

void ToolbarController::init(IntVector2D const & upperLeftPosition, DataManipulator* manipulator)
{
	_manipulator = manipulator;
	_view->init(upperLeftPosition, this);
	connect(_context, &ToolbarContext::show, this, &ToolbarController::onShow);

	onShow(false);
}

ToolbarContext * ToolbarController::getContext() const
{
	return _context;
}

void ToolbarController::onRequestCell()
{
	_manipulator->addAndSelectCell(_model->getPositionDeltaForNewEntity());
	_manipulator->reconnectSelectedCells();
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor
	});
}

void ToolbarController::onRequestParticle()
{
	_manipulator->addAndSelectParticle(_model->getPositionDeltaForNewEntity());
	Q_EMIT _manipulator->notify({
		DataManipulator::Receiver::DataEditor,
		DataManipulator::Receiver::Simulation,
		DataManipulator::Receiver::VisualEditor
	});
}

void ToolbarController::onShow(bool visible)
{
	_view->setVisible(visible);
}
