#ifndef CELLCLUSTERIMPL_H
#define CELLCLUSTERIMPL_H

#include "model/entities/cellcluster.h"

#include <QMatrix4x4>

class CellClusterImpl : public CellCluster
{
public:
    CellClusterImpl (SimulationContext* context);
    CellClusterImpl (QList< Cell* > cells, qreal angle, QVector3D pos, qreal angularVel, QVector3D vel, SimulationContext* context);
    CellClusterImpl (QList< Cell* > cells, qreal angle, SimulationContext* context);

    ~CellClusterImpl ();

    void clearCellsFromMap () override;
    void clearCellFromMap (Cell* cell) override;
    void drawCellsToMap () override;

    void processingInit () override;
    void processingDissipation (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles) override;
    void processingMovement () override;
    void processingToken (QList< EnergyParticle* >& energyParticles, bool& decompose) override;
    void processingFinish () override;

    void addCell (Cell* cell, QVector3D absPos) override;
    void removeCell (Cell* cell, bool maintainCenter = true) override;
    void updateCellVel (bool forceCheck = true) override;
    void updateAngularMass () override;
    void updateRelCoordinates (bool maintainCenter = false) override;
    void updateVel_angularVel_via_cellVelocities () override;
    QVector3D calcPosition (const Cell *cell, bool topologyCorrection = false) const override;
    QVector3D calcCellDistWithoutTorusCorrection (Cell* cell) const override;
    QList< CellCluster* > decompose () const override;
    qreal calcAngularMassWithNewParticle (QVector3D particlePos) const override;
    qreal calcAngularMassWithoutUpdate () const override;

    bool isEmpty() const override;

    QList< Cell* >& getCellsRef() override;
    const quint64& getId () const override;
    void setId (quint64 id) override;
    QList< quint64 > getCellIds () const override;
    QVector3D getPosition () const override;
    void setPosition (QVector3D pos, bool updateTransform = true) override;
    qreal getAngle () const override;
    void setAngle (qreal angle, bool updateTransform = true) override;
    QVector3D getVel () const override;
    void setVel (QVector3D vel) override;
    qreal getMass () const override;
    qreal getAngularVel () const override;
    void setAngularVel (qreal vel) override;
    qreal getAngularMass () const override;
    void updateTransformationMatrix () override;
    QVector3D relToAbsPos (QVector3D relPos) const override;
    QVector3D absToRelPos (QVector3D absPos) const override;

    void findNearestCells (QVector3D pos, Cell*& cell1, Cell*& cell2) const override;
    Cell* findNearestCell (QVector3D pos) const override;
    void getConnectedComponent(Cell* cell, QList< Cell* >& component) const override;
    void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const override;

	CellClusterMetadata getMetadata() const override;
	void setMetadata(CellClusterMetadata metadata) override;

    void serializePrimitives (QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

private:
    void radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle) const;

    SimulationContext* _context;
    Topology* _topology;
    CellMap* _cellMap;
    Grid* _grid;

    qreal _angle = 0.0;       //in deg
    QVector3D _pos;
    qreal _angularVel = 0.0;  //in deg
    QVector3D _vel;
    QMatrix4x4 _transform;
    qreal _angularMass = 0.0;

    QList< Cell* > _cells;

    quint64 _id = 0;

	CellClusterMetadata _meta;
};


#endif // CELLCLUSTERIMPL_H
