#ifndef COMPUTERCODEEDIT_H
#define COMPUTERCODEEDIT_H

#include <QTextEdit>

class ComputerCodeEdit : public QTextEdit
{
    Q_OBJECT
public:
    ComputerCodeEdit(QWidget *parent = 0);
    ~ComputerCodeEdit();

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

#endif // COMPUTERCODEEDIT_H
