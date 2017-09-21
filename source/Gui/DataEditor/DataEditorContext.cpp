#include "DataEditorContext.h"

DataEditorContext::DataEditorContext(DataEditorModel* model, QObject *parent /*= nullptr*/)
	: QObject(parent)
	, _model(model)
{

}

DataEditorModel* DataEditorContext::getModel()
{
	return _model;
}
