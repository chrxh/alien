#ifndef CLUSTEREDIT_H
#define CLUSTEREDIT_H

#include <QTextEdit>
#include <QVector3D>

#include "../../simulation/aliencellreduced.h"

class AlienCell;
class ClusterEdit : public QTextEdit
{
    Q_OBJECT
public:
    explicit ClusterEdit(QWidget *parent = 0);

    void updateCluster (AlienCellReduced cell);
    void requestUpdate ();

signals:
    void clusterDataChanged (AlienCellReduced cell);

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

    AlienCellReduced _cell;
};

#endif // CLUSTEREDIT_H
