#ifndef UNIVERSEPIXELSCENE_H
#define UNIVERSEPIXELSCENE_H

#include <QObject>

class AlienGrid;
class QGraphicsScene;
class QGraphicsPixmapItem;
class QImage;
class UniversePixelScene: public QObject
{
    Q_OBJECT
public:
    UniversePixelScene(QGraphicsScene* pixelScene, QObject* parent = 0);
    ~UniversePixelScene();

    void init (AlienGrid* space);
    void visualize ();

private:
    QGraphicsPixmapItem* _pixelMap;
    AlienGrid* _space;
    QImage* _image;
};

#endif // UNIVERSEPIXELSCENE_H
