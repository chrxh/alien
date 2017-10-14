#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "Model/Api/Descriptions.h"
#include "Model/Api/Definitions.h"
#include "Gui/Definitions.h"

class ClusterEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    ClusterEditWidget(QWidget *parent = 0);

	void init(DataEditorModel* model, DataEditorController* controller);
    void updateDisplay ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
	void requestUpdate();

    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};
