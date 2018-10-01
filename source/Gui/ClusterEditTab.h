#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "ModelInterface/Descriptions.h"
#include "ModelInterface/Definitions.h"
#include "Gui/Definitions.h"

class ClusterEditTab
	: public QTextEdit
{
    Q_OBJECT
public:
    ClusterEditTab(QWidget *parent = 0);

	void init(DataEditModel* model, DataEditController* controller);
    void updateDisplay ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
	void requestUpdate();

    qreal generateNumberFromFormattedString (QString s);

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
};
