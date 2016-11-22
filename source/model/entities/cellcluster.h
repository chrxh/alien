#ifndef CELLCLUSTER_H
#define CELLCLUSTER_H

#include <QList>

#include "grid.h"
#include "energyparticle.h"
#include <QMatrix4x4>

class Cell;

class CellCluster
{
public:
    static CellCluster* buildEmptyCellCluster (Grid*& grid);
    static CellCluster* buildCellCluster (QList< Cell* > cells,
                                               qreal angle,
                                               QVector3D pos,
                                               qreal angularVel,
                                               QVector3D vel,
                                               Grid*& grid);
    static CellCluster* buildCellCluster (QDataStream& stream,
                                               QMap< quint64, quint64 >& oldNewClusterIdMap,
                                               QMap< quint64, quint64 >& oldNewCellIdMap,
                                               QMap< quint64, Cell* >& oldIdCellMap,
                                               Grid*& grid);
    static CellCluster* buildCellClusterFromForeignCells (QList< Cell* > cells,
                                                               qreal angle,
                                                               Grid*& grid);

    ~CellCluster ();

    bool compareEqual (CellCluster* otherCluster) const;

    void clearCellsFromMap ();
    void clearCellFromMap (Cell* cell);
    void drawCellsToMap ();

    void movementProcessingStep1 ();
    void movementProcessingStep2 (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles);
    void movementProcessingStep3 ();
    void movementProcessingStep4 (QList< EnergyParticle* >& energyParticles, bool& decompose);
    void movementProcessingStep5 ();

    void addCell (Cell* cell, QVector3D absPos);
    void removeCell (Cell* cell, bool maintainCenter = true);
    void updateCellVel (bool forceCheck = true);        //forceCheck = true: large forces destroy cell
    void updateAngularMass ();
    void updateRelCoordinates (bool maintainCenter = false);
    void updateVel_angularVel_via_cellVelocities ();
    QVector3D calcPosition (const Cell *cell, bool topologyCorrection = false) const;
    QVector3D calcTopologyCorrection (CellCluster* cluster);
    QVector3D calcCellDistWithoutTorusCorrection (Cell* cell);
    QList< CellCluster* > decompose ();
    qreal calcAngularMassWithNewParticle (QVector3D particlePos);
    qreal calcAngularMassWithoutUpdate ();

    bool isEmpty();

    QList< Cell* >& getCells ();
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

    void findNearestCells (QVector3D pos, Cell*& cell1, Cell*& cell2);
    Cell* findNearestCell (QVector3D pos);
    void getConnectedComponent(Cell* cell, QList< Cell* >& component);
    void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component);

private:
    CellCluster (Grid*& grid);
    CellCluster (QList< Cell* > cells,
                      qreal angle,
                      QVector3D pos,
                      qreal angularVel,
                      QVector3D vel,
                      Grid*& grid);
    CellCluster (QDataStream& stream,
                      QMap< quint64, quint64 >& oldNewClusterIdMap,
                      QMap< quint64, quint64 >& oldNewCellIdMap,
                      QMap< quint64, Cell* >& oldIdCellMap,
                      Grid*& grid);
    CellCluster (QList< Cell* > cells,
                      qreal angle,
                      Grid*& grid);

    void radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle);

    Grid*& _grid;

    qreal _angle = 0.0;       //in deg
    QVector3D _pos;
    qreal _angularVel = 0.0;  //in deg
    QVector3D _vel;
    QMatrix4x4 _transform;
    qreal _angularMass = 0.0;

    QList< Cell* > _cells;

    quint64 _id = 0;
    quint64 _color = 0;
};


#endif // CELLCLUSTER_H
