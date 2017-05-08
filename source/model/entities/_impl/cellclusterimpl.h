#ifndef CELLCLUSTERIMPL_H
#define CELLCLUSTERIMPL_H

#include "model/entities/CellCluster.h"

#include <QMatrix4x4>

class CellClusterImpl
	: public CellCluster
{
public:
    CellClusterImpl (UnitContext* context);
    CellClusterImpl (QList< Cell* > cells, qreal angle, QVector2D pos, qreal angularVel, QVector2D vel, UnitContext* context);
    CellClusterImpl (QList< Cell* > cells, qreal angle, UnitContext* context);

    ~CellClusterImpl ();

	virtual void setContext(UnitContext* context) override;

	virtual CellClusterDescription getDescription() const override;

    void clearCellsFromMap () override;
    void clearCellFromMap (Cell* cell) override;
    void drawCellsToMap () override;

    void processingInit () override;
    void processingDissipation (QList< CellCluster* >& fragments, QList< EnergyParticle* >& energyParticles) override;
	void processingMutationByChance() override;
	void processingMovement() override;
    void processingToken (QList< EnergyParticle* >& energyParticles, bool& decompose) override;
    void processingCompletion () override;

    void addCell (Cell* cell, QVector2D absPos) override;
    void removeCell (Cell* cell, bool maintainCenter = true) override;
    void updateCellVel (bool forceCheck = true) override;
    void updateAngularMass () override;
    void updateRelCoordinates (bool maintainCenter = false) override;
    void updateVel_angularVel_via_cellVelocities () override;
    QVector2D calcPosition (const Cell *cell, bool metricCorrection = false) const override;
    QVector2D calcCellDistWithoutTorusCorrection (Cell* cell) const override;
    QList< CellCluster* > decompose () const override;
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

	CellClusterMetadata getMetadata() const override;
	void setMetadata(CellClusterMetadata metadata) override;

    void serializePrimitives (QDataStream& stream) const override;
	virtual void deserializePrimitives(QDataStream& stream) override;

private:
    void radiation (qreal& energy, Cell* originCell, EnergyParticle*& energyParticle) const;
	inline QVector2D applyTransformation(QVector2D pos) const;
	inline QVector2D applyTransformation(QMatrix4x4 const& transform, QVector2D pos) const;
	inline QVector2D applyInverseTransformation(QVector2D pos) const;

    UnitContext* _context = nullptr;

    qreal _angle = 0.0;       //in deg
    QVector2D _pos;
    qreal _angularVel = 0.0;  //in deg
    QVector2D _vel;
    QMatrix4x4 _transform;
    qreal _angularMass = 0.0;

    QList<Cell*> _cells;

    quint64 _id = 0;

	CellClusterMetadata _meta;
};

/********************* inline methods ******************/

QVector2D CellClusterImpl::applyTransformation(QVector2D pos) const
{
	return _transform.map(QVector3D(pos)).toVector2D();
}

QVector2D CellClusterImpl::applyTransformation(QMatrix4x4 const & transform, QVector2D pos) const
{
	return QVector2D();
}

QVector2D CellClusterImpl::applyInverseTransformation(QVector2D pos) const
{
	return _transform.inverted().map(QVector3D(pos)).toVector2D();
}

#endif // CELLCLUSTERIMPL_H
