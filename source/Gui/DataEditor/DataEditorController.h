#pragma once

#include <QWidget>
#include "Gui/Definitions.h"

class DataEditorController
	: public QObject
{
	Q_OBJECT
public:
	DataEditorController(QWidget *parent = nullptr);
	virtual ~DataEditorController() = default;

	void init(IntVector2D const& upperLeftPosition, DataManipulator* manipulator);

	DataEditorContext* getContext() const;

private:
	Q_SLOT void onShow(bool visible);
	Q_SLOT void dataUpdatedFromManipulator(set<UpdateTarget> const& argets);

	DataEditorView* _view = nullptr;
	DataManipulator* _manipulator = nullptr;
	DataEditorContext* _context = nullptr;
};