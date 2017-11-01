#pragma once

#include <QWidget>

#include "Model/Api/Definitions.h"
#include "Gui/Definitions.h"

namespace Ui {
class SymbolEditTab;
}

class SymbolEditTab
	: public QWidget
{
    Q_OBJECT
    
public:
    SymbolEditTab(QWidget *parent = nullptr);
    virtual ~SymbolEditTab();

	void init(DataEditorModel* model, DataEditorController* controller);
	void updateDisplay();

private:
    Q_SLOT void addSymbolButtonClicked ();
	Q_SLOT void delSymbolButtonClicked ();
	Q_SLOT void itemSelectionChanged ();
	Q_SLOT void itemContentChanged (QTableWidgetItem* item);

    Ui::SymbolEditTab *ui;

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};

