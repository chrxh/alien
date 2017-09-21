#include "DataEditorController.h"
#include "DataEditorContext.h"
#include "DataEditorModel.h"

DataEditorController::DataEditorController(QObject *parent /*= nullptr*/)
	: QObject(parent)
{
	_model = new DataEditorModel(this);
	_context = new DataEditorContext(_model, this);
}

DataEditorContext * DataEditorController::getContext() const
{
	return _context;
}
