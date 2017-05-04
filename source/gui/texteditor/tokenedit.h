#ifndef TOKENEDIT_H
#define TOKENEDIT_H

#include <QTextEdit>

class TokenEdit : public QTextEdit
{
    Q_OBJECT
public:
    TokenEdit (QWidget *parent = 0);
    ~TokenEdit ();

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

#endif // TOKENEDIT_H
