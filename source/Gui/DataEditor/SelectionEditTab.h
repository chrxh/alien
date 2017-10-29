#pragma once

#include <QTextEdit>

#include "Gui/Definitions.h"


class SelectionEditTab
	: public QTextEdit
{
	Q_OBJECT

public:
	SelectionEditTab(QWidget * parent = nullptr);
	virtual ~SelectionEditTab() = default;

	void init(DataEditorModel* model, DataEditorController* controller);
	void updateDisplay();

private:
	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;

};
