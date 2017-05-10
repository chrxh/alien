#ifndef CELLCLUSTER_H
#define CELLCLUSTER_H

#include <QList>
#include <QVector2D>

#include "model/Definitions.h"
#include "Descriptions.h"

class CellCluster
{
public:
	virtual ~CellCluster() = default;

	virtual void setContext(UnitContext* context) = 0;

	virtual CellClusterDescription getDescription(ResolveDescription const& resolveDescription) const = 0;

    virtual void clearCellsFromMap () = 0;
    virtual void clearCellFromMap (Cell* cell) = 0;
    virtual void drawCellsToMap () = 0;

    virtual void processingInit () = 0;
    virtual void processingDissipation (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles) = 0;
	virtual void processingMutationByChance() = 0;
	virtual void processingMovement() = 0;
    virtual void processingToken (QList< EnergyParticle* >& energyParticles, bool& decompose) = 0;
    virtual void processingCompletion () = 0;

    virtual void addCell (Cell* cell, QVector2D absPos) = 0;
    virtual void removeCell (Cell* cell, bool maintainCenter = true) = 0;
    virtual void updateCellVel (bool forceCheck = true) = 0;        //forceCheck = true: large forces destroy cell
    virtual void updateAngularMass () = 0;
    virtual void updateRelCoordinates (bool maintainCenter = false) = 0;
    virtual void updateVel_angularVel_via_cellVelocities () = 0;
    virtual QVector2D calcPosition (const Cell *cell, bool metricCorrection = false) const = 0;
    virtual QVector2D calcCellDistWithoutTorusCorrection (Cell* cell) const = 0;
    virtual QList< CellCluster* > decompose () const = 0;
    virtual qreal calcAngularMassWithNewParticle (QVector2D particlePos) const = 0;
    virtual qreal calcAngularMassWithoutUpdate () const = 0;

    virtual bool isEmpty() const = 0;

    virtual QList< Cell* >& getCellsRef () = 0;
    virtual const quint64& getId () const = 0;
    virtual void setId (quint64 id) = 0;
    virtual QList< quint64 > getCellIds () const = 0;
    virtual QVector2D getPosition () const = 0;
    virtual void setCenterPosition (QVector2D pos, bool updateTransform = true) = 0;
    virtual qreal getAngle () const = 0;  //in degrees
    virtual void setAngle (qreal angle, bool updateTransform = true) = 0;
    virtual QVector2D getVelocity () const = 0;
    virtual void setVelocity (QVector2D vel) = 0;
    virtual qreal getMass () const = 0;
    virtual qreal getAngularVel () const = 0;
    virtual void setAngularVel (qreal vel) = 0;
    virtual qreal getAngularMass () const = 0;
    virtual void updateTransformationMatrix () = 0;
    virtual QVector2D relToAbsPos (QVector2D relPos) const = 0;
    virtual QVector2D absToRelPos (QVector2D absPos) const = 0;

    virtual void findNearestCells (QVector2D pos, Cell*& cell1, Cell*& cell2) const = 0;
    virtual Cell* findNearestCell (QVector2D pos) const = 0;
    virtual void getConnectedComponent(Cell* cell, QList< Cell* >& component) const = 0;
    virtual void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const = 0;

	virtual CellClusterMetadata getMetadata() const = 0;
	virtual void setMetadata(CellClusterMetadata metadata) = 0;

    virtual void serializePrimitives (QDataStream& stream) const = 0;
	virtual void deserializePrimitives (QDataStream& stream) = 0;
};


#endif // CELLCLUSTER_H
