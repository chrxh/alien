#pragma once

#include <QTextEdit>

class CodeEditWidget : public QTextEdit
{
    Q_OBJECT
public:
    CodeEditWidget(QWidget *parent = 0);
    ~CodeEditWidget();

    void update (QString code);
    void update ();
    QString getCode ();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
    void displayData (QString code);
    void insertLineNumbers ();
    void removeLineNumbers ();
};
