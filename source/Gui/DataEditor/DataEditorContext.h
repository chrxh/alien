#pragma once

#include <QObject>

#include "Gui/Definitions.h"
#include "Model/Entities/Descriptions.h"

class DataEditorContext
	: public QObject
{
	Q_OBJECT
public:
	DataEditorContext(DataEditorModel* model, QObject *parent = nullptr);
	virtual ~DataEditorContext() = default;

	DataEditorModel* getModel();

	Q_SIGNAL void notifyDataEditor();
	Q_SIGNAL void notifyExternals();

private:
	DataEditorModel* _model;
};