#pragma once

#include <QTextEdit>
#include "Gui/Definitions.h"

class MetadataEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    MetadataEditWidget(QWidget *parent = 0);
	virtual ~MetadataEditWidget() = default;

	void init(DataEditModel* model, DataEditController* controller);
	void updateDisplay();

private:
    Q_SLOT void keyPressEvent (QKeyEvent* e);
	Q_SLOT void mousePressEvent (QMouseEvent* e);
	Q_SLOT void mouseDoubleClickEvent (QMouseEvent* e);

	void updateModel();

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
};
