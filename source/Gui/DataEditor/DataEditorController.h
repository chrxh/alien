#pragma once

#include <QWidget>
#include "Gui/Definitions.h"

class DataEditorController
	: public QObject
{
	Q_OBJECT
public:
	DataEditorController(IntVector2D const& upperLeftPosition, QWidget *parent = nullptr);
	virtual ~DataEditorController() = default;

	DataEditorContext* getContext() const;

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void notificationFromContext();

	DataEditorModel* _model = nullptr;
	DataEditorView* _view = nullptr;
	DataEditorContext* _context = nullptr;
};