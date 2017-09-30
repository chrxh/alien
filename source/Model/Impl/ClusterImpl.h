#pragma once

#include <QMatrix4x4>

#include "Model/Local/Cluster.h"

class ClusterImpl
	: public Cluster
{
public:
    ClusterImpl (QList< Cell* > cells, uint64_t id, qreal angle, QVector2D pos, qreal angularVel, QVector2D vel, UnitContext* context);

    ~ClusterImpl ();

	virtual void setContext(UnitContext* context) override;

	virtual ClusterDescription getDescription(ResolveDescription const& resolveDescription) const override;

    void clearCellsFromMap () override;
    void clearCellFromMap (Cell* cell) override;
    void drawCellsToMap () override;

    void processingInit () override;
    void processingDissipation (QList< Cluster* >& fragments, QList< Particle* >& energyParticles) override;
	void processingMutationByChance() override;
	void processingMovement() override;
    void processingToken (QList< Particle* >& energyParticles, bool& decompose) override;
    void processingCompletion () override;

    void addCell (Cell* cell, QVector2D absPos, UpdateInternals update = UpdateInternals::Yes) override;
	void removeCell(Cell* cell, MaintainCenter maintainCenter = MaintainCenter::Yes) override;
    void updateCellVel (bool forceCheck = true) override;
    void updateAngularMass () override;
    void updateRelCoordinates (MaintainCenter maintainCenter = MaintainCenter::No) override;
    void updateVel_angularVel_via_cellVelocities () override;
    QVector2D calcPosition (const Cell *cell, bool metricCorrection = false) const override;
    QVector2D calcCellDistWithoutTorusCorrection (Cell* cell) const override;
    QList< Cluster* > decompose () const override;
    qreal calcAngularMassWithNewParticle (QVector2D particlePos) const override;
    qreal calcAngularMassWithoutUpdate () const override;

    bool isEmpty() const override;

    QList< Cell* >& getCellsRef() override;
    const quint64& getId () const override;
    void setId (quint64 id) override;
    QList< quint64 > getCellIds () const override;
    QVector2D getPosition () const override;
    void setCenterPosition (QVector2D pos, bool updateTransform = true) override;
    qreal getAngle () const override;
    void setAngle (qreal angle, bool updateTransform = true) override;
    QVector2D getVelocity () const override;
    void setVelocity (QVector2D vel) override;
    qreal getMass () const override;
    qreal getAngularVel () const override;
    void setAngularVel (qreal vel) override;
    qreal getAngularMass () const override;
    void updateTransformationMatrix () override;
    QVector2D relToAbsPos (QVector2D relPos) const override;
    QVector2D absToRelPos (QVector2D absPos) const override;

    void findNearestCells (QVector2D pos, Cell*& cell1, Cell*& cell2) const override;
    Cell* findNearestCell (QVector2D pos) const override;
    void getConnectedComponent(Cell* cell, QList< Cell* >& component) const override;
    void getConnectedComponent(Cell* cell, const quint64& tag, QList< Cell* >& component) const override;

	ClusterMetadata getMetadata() const override;
	void setMetadata(ClusterMetadata metadata) override;

	void updateInternals(MaintainCenter maintanCenter = MaintainCenter::No) override;

    void serializePrimitives (QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

private:
	ClusterImpl(QList< Cell* > cells, qreal angle, UnitContext* context);

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

QVector2D ClusterImpl::applyTransformation(QVector2D pos) const
{
	return _transform.map(QVector3D(pos)).toVector2D();
}

QVector2D ClusterImpl::applyTransformation(QMatrix4x4 const & transform, QVector2D pos) const
{
	return QVector2D();
}

QVector2D ClusterImpl::applyInverseTransformation(QVector2D pos) const
{
	return _transform.inverted().map(QVector3D(pos)).toVector2D();
}
