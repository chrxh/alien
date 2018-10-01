#pragma once

#include <QTextEdit>

#include "ModelInterface/Definitions.h"
#include "Gui/Definitions.h"

class CodeEditWidget : public QTextEdit
{
    Q_OBJECT
public:
    CodeEditWidget(QWidget *parent = 0);
	virtual ~CodeEditWidget() = default;

	void init(DataEditModel* model, DataEditController* controller, CellComputerCompiler* compiler);
    void updateDisplay ();

    std::string getCode ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
    void displayData (QString code);
    void insertLineNumbers ();
    void removeLineNumbers ();

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
	CellComputerCompiler* _compiler = nullptr;
};
