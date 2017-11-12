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

	void init(DataEditModel* model, DataEditController* controller);
	void updateDisplay();

private:
	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;

};
