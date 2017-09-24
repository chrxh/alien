#include "Gui/DataManipulator.h"

#include "DataEditorController.h"
#include "DataEditorContext.h"
#include "DataEditorView.h"

DataEditorController::DataEditorController(QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_view = new DataEditorView(parent);
	_context = new DataEditorContext(this);
}

void DataEditorController::init(IntVector2D const & upperLeftPosition, DataManipulator * manipulator)
{
	_view->init(upperLeftPosition);
	_manipulator = manipulator;

	connect(_context, &DataEditorContext::show, this, &DataEditorController::onShow);
	connect(_manipulator, &DataManipulator::dataUpdated, this, &DataEditorController::dataUpdatedFromManipulator);

	onShow(false);
}

DataEditorContext * DataEditorController::getContext() const
{
	return _context;
}

void DataEditorController::onShow(bool visible)
{
	_view->show(visible);
}

void DataEditorController::dataUpdatedFromManipulator()
{
	_view->update();
}
