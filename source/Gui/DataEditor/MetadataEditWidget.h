#pragma once

#include <QWidget>
#include "Gui/Definitions.h"

namespace Ui {
class MetadataEditWidget;
}

class MetadataEditWidget
	: public QWidget
{
    Q_OBJECT

public:
    MetadataEditWidget(QWidget *parent = 0);
    virtual ~MetadataEditWidget();

	void init(DataEditorModel* model, DataEditorController* controller);
    void updateDisplay ();

private:
    Q_SLOT void changesFromMetadataDescriptionEditor ();

private:
    Ui::MetadataEditWidget *ui;

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};
