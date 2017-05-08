#ifndef PIXELUNIVERSE_H
#define PIXELUNIVERSE_H

#include <QGraphicsScene>
#include <QVector2D>
#include <QTimer>

#include "model/Definitions.h"

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

private Q_SLOTS:
    void timeout ();

private:
	SimulationContext* _context;
    QGraphicsPixmapItem* _pixelMap;
    QImage* _image;
    QTimer* _timer;

    QList< CellCluster* > _selectedClusters;
    QVector2D _selectionPos;
    QVector< QVector2D > _lastMouseDiffs;
    bool _leftMouseButtonPressed;
    bool _rightMouseButtonPressed;
};

#endif // PIXELUNIVERSE_H
