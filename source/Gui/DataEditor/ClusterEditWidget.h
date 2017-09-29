#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "Model/Entities/Descriptions.h"
#include "Model/Definitions.h"
#include "Gui/Definitions.h"

class ClusterEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    explicit ClusterEditWidget(QWidget *parent = 0);

	void init(DataEditorModel* model, DataEditorController* controller);
    void updateDisplay ();

	void requestUpdate();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:

    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};
