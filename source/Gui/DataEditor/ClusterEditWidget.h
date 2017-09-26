#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "Model/Entities/Descriptions.h"
#include "Model/Definitions.h"

class ClusterEditWidget
	: public QTextEdit
{
    Q_OBJECT
public:
    explicit ClusterEditWidget(QWidget *parent = 0);

    void updateCluster (ClusterDescription const& cluster);
    void requestUpdate ();

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

	ClusterDescription _cluster;
};
