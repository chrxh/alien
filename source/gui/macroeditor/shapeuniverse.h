#ifndef SHAPEUNIVERSE_H
#define SHAPEUNIVERSE_H

#include "model/aliencellreduced.h"

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector3D>

class AlienCell;
class AlienCellCluster;
class AlienCellGraphicsItem;
class AlienCellConnectionGraphicsItem;
class AlienEnergy;
class AlienEnergyGraphicsItem;
class AlienGrid;
class MarkerGraphicsItem;
class QGraphicsSceneMouseEvent;
class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = 0);

    void universeUpdated (AlienGrid* grid);
    void cellCreated (AlienCell* cell);
    void energyParticleCreated (AlienEnergy* e);
    void defocused ();
    void energyParticleUpdated_Slot (AlienEnergy* e);
    void getExtendedSelection (QList< AlienCellCluster* >& clusters, QList< AlienEnergy* >& es);
    void delSelection (QList< AlienCell* >& cells, QList< AlienEnergy* >& es);
    void delExtendedSelection (QList< AlienCellCluster* >& clusters, QList< AlienEnergy* >& es);
    void metadataUpdated ();
    QGraphicsItem* getFocusCenterCell ();

public slots:
    void reclustered (QList< AlienCellCluster* > clusters);

signals:
    void defocus ();                                                //to microeditor
    void focusCell (AlienCell* cell);                               //to microeditor
    void focusEnergyParticle (AlienEnergy* e);                      //to microeditor
    void focusEnsemble (int numCells, int numEnergyParticles);      //to microeditor
    void updateCell (QList< AlienCell* > cells,
                     QList< AlienCellReduced > newCellsData,
                     bool clusterDataChanged);                      //to simulator
    void energyParticleUpdated (AlienEnergy* e);                    //to microeditor
    void entitiesSelected (int numCells, int numEnergyParticles);   //to microeditor

protected:

    //events
    void mousePressEvent (QGraphicsSceneMouseEvent* e);
    void mouseReleaseEvent (QGraphicsSceneMouseEvent* e);
    void mouseMoveEvent (QGraphicsSceneMouseEvent* e);


private:

    //internal methods
    AlienEnergyGraphicsItem* createEnergyItem (AlienEnergy* e);
    AlienCellGraphicsItem* createCellItem (AlienCell* cell);
    void createConnectionItem (AlienCell* cell, AlienCell* otherCell);
    void delConnectionItem (quint64 cellId);
    void unhighlight ();
    void highlightCell (AlienCell* cell);
    void highlightEnergyParticle (AlienEnergyGraphicsItem* e);
    void setCellColorFromMetadata ();

    //internal data for display and editing
    AlienGrid* _grid;
    QList< AlienCellGraphicsItem* > _focusCells;
    QList< AlienEnergyGraphicsItem* > _focusEnergyParticles;
    QMap< quint64, AlienCellGraphicsItem* > _highlightedCells;  //contains the whole clusters when a single cell in focused therein
    QMap< quint64, AlienEnergyGraphicsItem* > _highlightedEnergyParticles;
    MarkerGraphicsItem* _marker;
    AlienCellGraphicsItem* _focusCenterCellItem;

    //associations
    QMap< quint64, AlienCellGraphicsItem* > _cellItems;
    QMap< quint64, AlienEnergyGraphicsItem* > _energyItems;
    QMap< quint64, QMap< quint64, AlienCellConnectionGraphicsItem* > > _connectionItems;

    QVector3D calcCenterOfHighlightedObjects ();
};

#endif // SHAPEUNIVERSE_H
