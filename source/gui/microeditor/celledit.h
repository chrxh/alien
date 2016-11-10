#ifndef CELLEDIT_H
#define CELLEDIT_H

#include <QTextEdit>
#include <QVector3D>

#include "model/entities/cellto.h"

class Cell;
class CellEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit CellEdit(QWidget *parent = 0);

    void updateCell (CellTO cell);
    void requestUpdate ();

signals:
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
    QString generateFormattedCellFunctionString (CellFunctionType type);

    CellTO _cell;
};

#endif // CELLEDIT_H
