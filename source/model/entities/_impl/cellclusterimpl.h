#ifndef CELLCLUSTERIMPL_H
#define CELLCLUSTERIMPL_H

#include "model/entities/cellcluster.h"

#include <QMatrix4x4>

class CellClusterImpl : public CellCluster
{
public:
    CellClusterImpl (Grid* grid);
    CellClusterImpl (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel, Grid* grid);
    CellClusterImpl (QList< Cell* > cells, qreal angle, Grid* grid);

    ~CellClusterImpl ();

    void clearCellsFromMap ();
    void clearCellFromMap (Cell* cell);
    void drawCellsToMap ();

    void processingInit ();
    void processingDissipation (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles);
    void processingMovement ();
    void processingToken (QList< EnergyParticle* >& energyParticles, bool& decompose);
    void processingFinish ();

    void addCell (Cell* cell, QVector3D absPos);
    void removeCell (Cell* cell, bool maintainCenter = true);
    void updateCellVel (bool forceCheck = true);
    void updateAngularMass ();
    void updateRelCoordinates (bool maintainCenter = false);
    void updateVel_angularVel_via_cellVelocities ();
    QVector3D calcPosition (const Cell *cell, bool topologyCorrection = false) const;
    QVector3D calcTopologyCorrection (CellCluster* cluster) const;
    QVector3D calcCellDistWithoutTorusCorrection (Cell* cell) const;
    QList< CellCluster* > decompose () const;
    qreal calcAngularMassWithNewParticle (QVector3D particlePos) const;
    qreal calcAngularMassWithoutUpdate () const;

    bool isEmpty() const;

    QList< Cell* >& getCellsRef();
    const quint64& getId () const;
    void setId (quint64 id);
    QList< quint64 > getCellIds () const;
    quint64 getColor () const;
    void setColor (quint64 color);
    QVector3D getPosition () const;
    void setPosition (QVector3D pos, bool updateTransform = true);
    qreal getAngle () const;
    void setAngle (qreal angle, bool updateTransform = true);
    QVector3D getVel () const;
    void setVel (QVector3D vel);
    qreal getMass () const;
    qreal getAngularVel () const;
    void setAngularVel (qreal vel);
    qreal getAngularMass () const;
    void updateTransformationMatrix ();
    QVector3D relToAbsPos (QVector3D relPos) const;
    QVector3D absToRelPos (QVector3D absPos) const;

    void findNearestCells (QVector3D pos, Cell*& cell1, Cell*& cell2) const;
    Cell* findNearestCell (QVector3D pos) const;
    void getConnectedComponent(Cell* cell, QList< Cell* >& component) const;
    void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const;

    void serializePrimitives (QDataStream& stream) const;
	virtual void deserializePrimitives(QDataStream& stream);

private:
    void radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle) const;

    Grid* _grid;

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


#endif // CELLCLUSTERIMPL_H
