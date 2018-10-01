#pragma once

#include <QWidget>

#include "ModelInterface/Definitions.h"
#include "Gui/Definitions.h"

namespace Ui {
class CellComputerEditTab;
}

class CellComputerEditTab : public QWidget
{
    Q_OBJECT
    
public:
    CellComputerEditTab(QWidget *parent = 0);
    virtual ~CellComputerEditTab();

	void init(DataEditModel* model, DataEditController* controller, CellComputerCompiler* compiler);
	void updateDisplay();

private:
    Q_SLOT void compileButtonClicked ();
	Q_SLOT void timerTimeout ();
	Q_SLOT void updateFromMemoryEditWidget();
    
	void setCompilationState(bool error, int line);

    Ui::CellComputerEditTab *ui;
	QTimer* _timer = nullptr;
	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};
