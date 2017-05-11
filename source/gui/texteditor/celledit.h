#ifndef CELLEDIT_H
#define CELLEDIT_H

#include <QTextEdit>
#include <QVector2D>

#include "model/Entities/CellTO.h"

class Cell;
class CellEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit CellEdit(QWidget *parent = 0);

    void updateCell (CellTO cell);
    void requestUpdate ();

Q_SIGNALS:
    void cellDataChanged (CellTO cell);

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:

    void updateDisplay ();

    qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);
    QString generateFormattedCellFunctionString (Enums::CellFunction::Type type);

    CellTO _cell;
};

#endif // CELLEDIT_H
