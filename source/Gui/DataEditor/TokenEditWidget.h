#pragma once

#include <QTextEdit>

#include "Gui/Definitions.h"

class TokenEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    TokenEditWidget (QWidget *parent = nullptr);
	virtual ~TokenEditWidget() = default;

	void init(DataEditModel* model, DataEditController* controller, int tokenIndex);

    void updateDisplay();
    void requestUpdate ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (qreal r);

	DataEditModel* _model = nullptr;
	DataEditController* _controller = nullptr;
	int _tokenIndex = 0;
};
