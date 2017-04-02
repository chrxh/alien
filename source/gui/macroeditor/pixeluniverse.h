#ifndef PIXELUNIVERSE_H
#define PIXELUNIVERSE_H

#include <QGraphicsScene>
#include <QVector3D>
#include <QTimer>

#include "model/definitions.h"

class PixelUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    PixelUniverse(QObject* parent=0);
    ~PixelUniverse();

    void reset ();
    void universeUpdated (SimulationContext* context);

protected:

    //events
    void mousePressEvent (QGraphicsSceneMouseEvent* e);
    void mouseReleaseEvent (QGraphicsSceneMouseEvent* e);
    void mouseMoveEvent (QGraphicsSceneMouseEvent* e);

private slots:
    void timeout ();

private:
	SimulationContext* _context;
    QGraphicsPixmapItem* _pixelMap;
    QImage* _image;
    QTimer* _timer;

    QList< CellCluster* > _selectedClusters;
    QVector3D _selectionPos;
    QVector< QVector3D > _lastMouseDiffs;
    bool _leftMouseButtonPressed;
    bool _rightMouseButtonPressed;
};

#endif // PIXELUNIVERSE_H
