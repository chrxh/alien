#pragma once

#include <QList>
#include <QMatrix4x4>
#include <QVector2D>

#include "Model/Api/Definitions.h"
#include "Model/Api/ChangeDescriptions.h"
#include "EntityWithTimestamp.h"


class Cluster
	: public EntityWithTimestamp
{
public:
    Cluster (QList< Cell* > cells, uint64_t id, qreal angle, QVector2D pos, qreal angularVel, QVector2D vel, UnitContext* context);

    virtual ~Cluster ();

	virtual void setContext(UnitContext* context);

	virtual ClusterDescription getDescription(ResolveDescription const& resolveDescription) const;
	virtual void applyChangeDescription(ClusterChangeDescription const& change);

    void clearCellsFromMap ();
    void clearCellFromMap (Cell* cell);
    void drawCellsToMap ();

    void processingInit ();
    void processingDissipation (QList< Cluster* >& fragments, QList< Particle* >& energyParticles);
	void processingMutationByChance();
	void processingMovement();
    void processingToken (QList< Particle* >& energyParticles, bool& decompose);
    void processingCompletion ();

	enum class UpdateInternals { No, Yes };
	void addCell(Cell* cell, QVector2D absPos, UpdateInternals update = UpdateInternals::Yes);
	enum MaintainCenter { No, Yes };
	void removeCell(Cell* cell, MaintainCenter maintainCenter = MaintainCenter::Yes);
    void updateCellVel (bool forceCheck = true);
    void updateAngularMass ();
    void updateRelCoordinates (MaintainCenter maintainCenter = MaintainCenter::No);
    void updateVel_angularVel_via_cellVelocities ();
    QVector2D calcPosition (const Cell *cell, bool metricCorrection = false) const;
    QVector2D calcCellDistWithoutTorusCorrection (Cell* cell) const;
    QList< Cluster* > decompose () const;
    qreal calcAngularMassWithNewParticle (QVector2D particlePos) const;
    qreal calcAngularMassWithoutUpdate () const;

    bool isEmpty() const;

    QList< Cell* >& getCellsRef();
    const quint64& getId () const;
    void setId (quint64 id);
    QList< quint64 > getCellIds () const;
    QVector2D getPosition () const;
    void setCenterPosition (QVector2D pos, bool updateTransform = true);
    qreal getAngle () const;
    void setAngle (qreal angle, bool updateTransform = true);
    QVector2D getVelocity () const;
    void setVelocity (QVector2D vel);
    qreal getMass () const;
    qreal getAngularVel () const;
    void setAngularVel (qreal vel);
    qreal getAngularMass () const;
    void updateTransformationMatrix ();
    QVector2D relToAbsPos (QVector2D relPos) const;
    QVector2D absToRelPos (QVector2D absPos) const;

    void findNearestCells (QVector2D pos, Cell*& cell1, Cell*& cell2) const;
    Cell* findNearestCell (QVector2D pos) const;
    void getConnectedComponent(Cell* cell, QList< Cell* >& component) const;
    void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const;

	ClusterMetadata getMetadata() const;
	void setMetadata(ClusterMetadata metadata);

	void updateInternals(MaintainCenter maintanCenter = MaintainCenter::No);

    void serializePrimitives (QDataStream& stream) const;
	virtual void deserializePrimitives(QDataStream& stream);

private:
	Cluster(QList< Cell* > cells, qreal angle, UnitContext* context);

	void radiation (qreal& energy, Cell* originCell, Particle*& energyParticle) const;
	inline QVector2D applyTransformation(QVector2D pos) const;
	inline QVector2D applyTransformation(QMatrix4x4 const& transform, QVector2D pos) const;
	inline QVector2D applyInverseTransformation(QVector2D pos) const;

    qreal _angle = 0.0;       //in deg
    QVector2D _pos;
    qreal _angularVel = 0.0;  //in deg
    QVector2D _vel;
    QMatrix4x4 _transform;
    qreal _angularMass = 0.0;
	
    QList<Cell*> _cells;
	
    quint64 _id = 0;
	ClusterMetadata _meta;
};

/********************* inline methods ******************/

QVector2D Cluster::applyTransformation(QVector2D pos) const
{
	return _transform.map(QVector3D(pos)).toVector2D();
}

QVector2D Cluster::applyTransformation(QMatrix4x4 const & transform, QVector2D pos) const
{
	return QVector2D();
}

QVector2D Cluster::applyInverseTransformation(QVector2D pos) const
{
	return _transform.inverted().map(QVector3D(pos)).toVector2D();
}
