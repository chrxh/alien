#include "DataEditorController.h"
#include "DataEditorContext.h"

DataEditorController::DataEditorController(QObject *parent /*= nullptr*/)
	: QObject(parent)
{
	_context = new DataEditorContext(this);
}

DataEditorContext * DataEditorController::getContext() const
{
	return _context;
}
