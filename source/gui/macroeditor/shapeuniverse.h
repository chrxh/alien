#ifndef SHAPEUNIVERSE_H
#define SHAPEUNIVERSE_H

#include "model/entities/cellto.h"

#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QMap>
#include <QVector3D>

class Cell;
class CellCluster;
class CellGraphicsItem;
class CellConnectionGraphicsItem;
class EnergyParticle;
class EnergyGraphicsItem;
class Grid;
class MarkerGraphicsItem;
class QGraphicsSceneMouseEvent;
class ShapeUniverse : public QGraphicsScene
{
    Q_OBJECT
public:
    ShapeUniverse (QObject *parent = 0);

    void universeUpdated (Grid* grid);
    void cellCreated (Cell* cell);
    void energyParticleCreated (EnergyParticle* e);
    void defocused ();
    void energyParticleUpdated_Slot (EnergyParticle* e);
    void getExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es);
    void delSelection (QList< Cell* >& cells, QList< EnergyParticle* >& es);
    void delExtendedSelection (QList< CellCluster* >& clusters, QList< EnergyParticle* >& es);
    void metadataUpdated ();
    QGraphicsItem* getFocusCenterCell ();

public slots:
    void reclustered (QList< CellCluster* > clusters);

signals:
    void defocus ();                                                //to microeditor
    void focusCell (Cell* cell);                               //to microeditor
    void focusEnergyParticle (EnergyParticle* e);                      //to microeditor
    void focusEnsemble (int numCells, int numEnergyParticles);      //to microeditor
    void updateCell (QList< Cell* > cells,
                     QList< CellTO > newCellsData,
                     bool clusterDataChanged);                      //to simulator
    void energyParticleUpdated (EnergyParticle* e);                    //to microeditor
    void entitiesSelected (int numCells, int numEnergyParticles);   //to microeditor

protected:

    //events
    void mousePressEvent (QGraphicsSceneMouseEvent* e);
    void mouseReleaseEvent (QGraphicsSceneMouseEvent* e);
    void mouseMoveEvent (QGraphicsSceneMouseEvent* e);


private:

    //internal methods
    EnergyGraphicsItem* createEnergyItem (EnergyParticle* e);
    CellGraphicsItem* createCellItem (Cell* cell);
    void createConnectionItem (Cell* cell, Cell* otherCell);
    void delConnectionItem (quint64 cellId);
    void unhighlight ();
    void highlightCell (Cell* cell);
    void highlightEnergyParticle (EnergyGraphicsItem* e);
    void setCellColorFromMetadata ();

    //internal data for display and editing
    Grid* _grid;
    QList< CellGraphicsItem* > _focusCells;
    QList< EnergyGraphicsItem* > _focusEnergyParticles;
    QMap< quint64, CellGraphicsItem* > _highlightedCells;  //contains the whole clusters when a single cell in focused therein
    QMap< quint64, EnergyGraphicsItem* > _highlightedEnergyParticles;
    MarkerGraphicsItem* _marker;
    CellGraphicsItem* _focusCenterCellItem;

    //associations
    QMap< quint64, CellGraphicsItem* > _cellItems;
    QMap< quint64, EnergyGraphicsItem* > _energyItems;
    QMap< quint64, QMap< quint64, CellConnectionGraphicsItem* > > _connectionItems;

    QVector3D calcCenterOfHighlightedObjects ();
};

#endif // SHAPEUNIVERSE_H
