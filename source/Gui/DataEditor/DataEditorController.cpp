#include "DataEditorController.h"
#include "DataEditorContext.h"
#include "DataEditorModel.h"
#include "DataEditorView.h"

DataEditorController::DataEditorController(IntVector2D const& upperLeftPosition, QWidget *parent /*= nullptr*/)
	: QObject(parent)
{
	_model = new DataEditorModel(this);
	_view = new DataEditorView(upperLeftPosition, _model, parent);
	_context = new DataEditorContext(_model, this);

	connect(_context, &DataEditorContext::show, this, &DataEditorController::onShow);
	connect(_context, &DataEditorContext::notifyDataEditor, this, &DataEditorController::notificationFromContext);

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

void DataEditorController::notificationFromContext()
{
	_view->update();
}
