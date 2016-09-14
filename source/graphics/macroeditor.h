#ifndef MACROEDITOR_H
#define MACROEDITOR_H

#include "../simulation/aliencellreduced.h"

#include <QWidget>
#include <QVector3D>
#include <QMatrix>

namespace Ui {
class MacroEditor;
}

class AlienCell;
class AlienCellCluster;
class AlienEnergy;
class AlienGrid;
class MetaDataManager;
class PixelUniverse;
class ShapeUniverse;
class MacroEditor : public QWidget
{
    Q_OBJECT

public:
    enum ActiveScene {
        PIXEL_SCENE,
        SHAPE_SCENE
    };

    explicit MacroEditor(QWidget *parent = 0);
    ~MacroEditor();

    void init (MetaDataManager* meta);
    void reset ();

    void setActiveScene (ActiveScene activeScene);
    QVector3D getViewCenterPosWithInc ();
    void getExtendedSelection (QList< AlienCellCluster* >& clusters, QList< AlienEnergy* >& es);

    void serializeViewMatrix (QDataStream& stream);
    void loadViewMatrix (QDataStream& stream);

    qreal getZoomFactor ();

signals:
    void requestNewCell (QVector3D pos);    //for simulator
    void requestNewEnergyParticle (QVector3D pos);    //for simulator
    void defocus ();                        //for microeditor
    void focusCell (AlienCell* cell);       //for microeditor
    void focusEnergyParticle (AlienEnergy* e);       //for microeditor
    void updateCell (QList< AlienCell* > cells,
                     QList< AlienCellReduced > newCellsData,
                     bool clusterDataChanged);      //for simulator
    void energyParticleUpdated (AlienEnergy* e); //for microeditor
    void entitiesSelected (int numCells, int numEnergyParticles);   //for microeditor
    void delSelection (QList< AlienCell* > cells,
                      QList< AlienEnergy* > es);               //for simulator
    void delExtendedSelection (QList< AlienCellCluster* > clusters,
                         QList< AlienEnergy* > es);               //for simulator


public slots:
    void zoomIn ();
    void zoomOut ();

    void newCellRequested ();
    void newEnergyParticleRequested ();

    void defocused ();
    void delSelection_Slot ();
    void delExtendedSelection_Slot ();
    void cellCreated (AlienCell* cell);
    void energyParticleCreated (AlienEnergy* e);
    void energyParticleUpdated_Slot (AlienEnergy* e);
    void reclustered (QList< AlienCellCluster* > clusters);
    void universeUpdated (AlienGrid* grid, bool force);
    void metaDataUpdated ();

private slots:
    void updateTimerTimeout ();

private:
    void centerView (AlienGrid* grid);

    Ui::MacroEditor *ui;

    AlienGrid* _grid;
    ActiveScene _activeScene;
    PixelUniverse* _pixelUniverse;
    ShapeUniverse* _shapeUniverse;
    MetaDataManager* _meta;

    bool _pixelUniverseInit;
    bool _shapeUniverseInit;
    QMatrix _pixelUniverseViewMatrix;
    QMatrix _shapeUniverseViewMatrix;
    int _pixelUniversePosX, _pixelUniversePosY;
    int _shapeUniversePosX, _shapeUniversePosY;

    qreal _posIncrement;

    //timer for limiting screen updates
    QTimer* _updateTimer;
    bool _screenUpdatePossible;
};

#endif // MACROEDITOR_H









