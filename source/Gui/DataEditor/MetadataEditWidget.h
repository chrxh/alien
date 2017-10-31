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

	void init(DataEditorModel* model, DataEditorController* controller);
	void updateDisplay();

private:
    Q_SLOT void keyPressEvent (QKeyEvent* e);
	Q_SLOT void mousePressEvent (QMouseEvent* e);
	Q_SLOT void mouseDoubleClickEvent (QMouseEvent* e);

	void updateModel();

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};
