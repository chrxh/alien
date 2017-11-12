#pragma once

#include <QWidget>
#include "Gui/Definitions.h"

namespace Ui {
class MetadataEditTab;
}

class MetadataEditTab
	: public QWidget
{
    Q_OBJECT

public:
    MetadataEditTab(QWidget *parent = 0);
    virtual ~MetadataEditTab();

	void init(DataEditModel* model, DataEditController* controller);
    void updateDisplay ();

private:
    Q_SLOT void changesFromMetadataDescriptionEditor ();

private:
    Ui::MetadataEditTab *ui;

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
};
