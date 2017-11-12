#pragma once

#include <QTextEdit>

class TokenEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    TokenEditWidget (QWidget *parent = 0);
	virtual ~TokenEditWidget() = default;

    void update (qreal energy);
    void requestUpdate ();

Q_SIGNALS:
    void dataChanged (qreal energy);

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (qreal r);
};
