#ifndef ALIENCELLCLUSTER_H
#define ALIENCELLCLUSTER_H

#include <QList>

#include "aliencell.h"
#include "aliengrid.h"
#include "alienenergy.h"
#include <QMatrix4x4>


//QVector3D calcDistance (AlienCell* cell1, AlienCell* cell2);

class AlienCellCluster
{
public:
    AlienCellCluster (AlienGrid*& grid);
    AlienCellCluster (QList< AlienCell* > cells,
                      qreal angle,
                      QVector3D pos,
                      qreal angularVel,
                      QVector3D vel,
                      AlienGrid*& grid);   //create cluster with new cells
    AlienCellCluster (QList< AlienCell* > cells,
                      qreal angle,
                      AlienGrid*& grid);   //new cluster takes ownership of cells
    AlienCellCluster (QDataStream& stream,
                      QMap< quint64, quint64 >& oldNewClusterIdMap,
                      QMap< quint64, quint64 >& oldNewCellIdMap,
                      QMap< quint64, AlienCell* >& oldIdCellMap,
                      AlienGrid*& grid);
    ~AlienCellCluster ();

    void clearCellsFromMap ();
    void clearCellFromMap (AlienCell* cell);
    void drawCellsToMap ();

    void movementProcessingStep1 ();
    void movementProcessingStep2 (QList< AlienCellCluster* >& fragments, QList< AlienEnergy* >& energyParticles);
    void movementProcessingStep3 ();
    void movementProcessingStep4 (QList< AlienEnergy* >& energyParticles, bool& decompose);
    void movementProcessingStep5 ();

    void addCell (AlienCell* cell, QVector3D absPos);
    void removeCell (AlienCell* cell, bool maintainCenter = true);
    void updateCellVel (bool forceCheck = true);        //forceCheck = true: large forces destroy cell
    void updateAngularMass ();
    void updateRelCoordinates (bool maintainCenter = false);
    void updateVel_angularVel_via_cellVelocities ();
    QVector3D calcPosition (AlienCell* cell, bool topologyCorrection = false);
    QVector3D calcTorusCorrection (AlienCellCluster* cluster);
    QVector3D calcCellDistWithoutTorusCorrection (AlienCell* cell);
    QList< AlienCellCluster* > decompose ();
    qreal calcAngularMassWithNewParticle (QVector3D particlePos);
    qreal calcAngularMassWithoutUpdate ();

    bool isEmpty();

    QList< AlienCell* >& getCells ();
    const quint64& getId ();
    void setId (quint64 id);
    QList< quint64 > getCellIds ();
    const quint64& getColor ();
    QVector3D getPosition ();
    void setPosition (QVector3D pos, bool updateTransform = true);
    qreal getAngle ();  //in degrees
    void setAngle (qreal angle, bool updateTransform = true);
    QVector3D getVel ();
    void setVel (QVector3D vel);
    qreal getMass ();
    qreal getAngularVel ();
    void setAngularVel (qreal vel);
    qreal getAngularMass ();
    void calcTransform ();
    QVector3D relToAbsPos (QVector3D relPos);
    QVector3D absToRelPos (QVector3D absPos);

    void serialize (QDataStream& stream);

    void findNearestCells (QVector3D pos, AlienCell*& cell1, AlienCell*& cell2);
    AlienCell* findNearestCell (QVector3D pos);
    void getConnectedComponent(AlienCell* cell, QList< AlienCell* >& component);
    void getConnectedComponent(AlienCell* cell, const quint64& tag, QList< AlienCell* >& component);

private:
    void radiation (qreal& energy, AlienCell* originCell, AlienEnergy*& energyParticle);

    AlienGrid*& _grid;

    //physics data
    qreal _angle;       //in deg
    QVector3D _pos;

    qreal _angularVel;  //in deg
    QVector3D _vel;
    QMatrix4x4 _transform;

    qreal _angularMass;

    //cells
    QList< AlienCell* > _cells;

    quint64 _id;
    quint64 _color;
//    int _counter;
//    int debug;

    struct CollisionData {
        int movementState;  //0: will do nothing, 1: collision, 2: fusion
        QSet< quint64 > overlappingCells;
        QList< QPair< AlienCell*, AlienCell* > > overlappingCellPairs;
    };
};


#endif // ALIENCELLCLUSTER_H
