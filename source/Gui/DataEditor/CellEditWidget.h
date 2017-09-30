#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "Model/Definitions.h"
#include "Model/Descriptions.h"
#include "Gui/Definitions.h"

class CellEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    CellEditWidget(QWidget *parent = 0);
	virtual ~CellEditWidget() = default;

	void init(DataEditorModel* model, DataEditorController* controller);
    void updateDisplay ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
    void updateData ();

    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);
    QString generateFormattedCellFunctionString (Enums::CellFunction::Type type);

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};

