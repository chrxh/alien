#pragma once

#include <QWidget>

#include "Model/Api/Definitions.h"
#include "Gui/Definitions.h"

namespace Ui {
class CellComputerEditWidget;
}

class CellComputerEditWidget : public QWidget
{
    Q_OBJECT
    
public:
    CellComputerEditWidget(QWidget *parent = 0);
    virtual ~CellComputerEditWidget();

	void init(DataEditorModel* model, DataEditorController* controller, CellComputerCompiler* compiler);
	void updateDisplay();

private:
    Q_SLOT void compileButtonClicked ();
	Q_SLOT void timerTimeout ();
	Q_SLOT void updateFromMemoryEditWidget();
    
	void setCompilationState(bool error, int line);

    Ui::CellComputerEditWidget *ui;
	QTimer* _timer = nullptr;
	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};
