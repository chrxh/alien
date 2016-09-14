#ifndef UNIVERSESHAPESCENE_H
#define UNIVERSESHAPESCENE_H

#include "../../simulation/entities/aliengrid.h"
#include <QObject>
#include <QList>
#include <QMap>
#include <QVector3D>

class AlienCellCluster;
class AlienCell;
class AlienEnergy;
class AlienSimulator;
class QGraphicsScene;
class QGraphicsEllipseItem;
class QGraphicsLineItem;

struct ConnectionItem
{
    QGraphicsLineItem* line;
    QGraphicsLineItem* lineStart1;
    QGraphicsLineItem* lineStart2;
    QGraphicsLineItem* lineEnd1;
    QGraphicsLineItem* lineEnd2;

    ConnectionItem(QGraphicsLineItem* line_ = 0) : line(line_), lineStart1(0), lineStart2(0), lineEnd1(0), lineEnd2(0) {}
};

class UniverseShapeScene : public QObject
{
    Q_OBJECT
public:
    UniverseShapeScene(QGraphicsScene* editorScene, QObject* parent = 0);

    void init (AlienSimulator* simulator, AlienGrid* space);
    void visualize ();
    void isolateCell (AlienCell* cell);
    void updateEnergyParticle (AlienEnergy* energy);
    void updateCluster (AlienCellCluster* cluster);
    void updateFocusCell (AlienCell* cell);
    void updateToken ();
    void focusCell (AlienCell* cell);
    void focusEnergyParticle (AlienEnergy* energy);
    QVector3D getViewCenter ();
    void colorCluster (AlienCellCluster* cluster);
    void decolorCluster (QList< AlienCell* >& cells);

signals:
    void defocus ();
    void cellClicked (AlienCell* cell);
    void energyParticleClicked (AlienEnergy* energy);
    void energyParticleMoved ();
    void cellMoved ();
    void clusterMoved ();

private:
    QGraphicsScene* _editorScene;
    AlienSimulator* _simulator;
    AlienGrid* _space;
    QMap< quint64, QGraphicsEllipseItem* > _cellItems;
    QMap< quint64, QGraphicsEllipseItem* > _tokenItems;
    QMap< quint64, QMap< quint64, ConnectionItem > > _connectionItems;

    int _zoom;
    QGraphicsEllipseItem* _focusItem;
    QVector3D _focusRelPos;
    bool _focusItemMovable;
    qreal _lastVertMousePos;

    bool eventFilter(QObject *obj, QEvent *event);
    void drawArrow (QVector3D p1, QVector3D p2, QGraphicsLineItem*& line1, QGraphicsLineItem*& line2);
    void moveArrow (QVector3D p1, QVector3D p2, QGraphicsLineItem* line1, QGraphicsLineItem* line2);
    ConnectionItem createConnectionItem (AlienCell* cell, AlienCell* otherCell);
    void moveConnectionItem (AlienCell* cell, AlienCell* otherCell, ConnectionItem item);
};

#endif // UNIVERSESHAPESCENE_H
