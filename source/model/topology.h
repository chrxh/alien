#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "definitions.h"

class Topology
{
public:

	virtual ~Topology() {}

	virtual void init(IntVector2D size) = 0;

	virtual IntVector2D getSize() const = 0;

	virtual void correctPosition(QVector3D& pos) const = 0;
	virtual IntVector2D correctPositionWithIntPrecision(QVector3D const& pos) const = 0;
	virtual IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const = 0;
	virtual void correctDisplacement(QVector3D& displacement) const = 0;
	virtual QVector3D displacement(QVector3D fromPoint, QVector3D toPoint) const = 0;
	virtual qreal distance(QVector3D fromPoint, QVector3D toPoint) const = 0;
	virtual QVector3D correctionIncrement (QVector3D pos1, QVector3D pos2) const = 0;

	virtual void serializePrimitives(QDataStream& stream) const = 0;
	virtual void deserializePrimitives(QDataStream& stream) = 0;
};

#endif // TOPOLOGY_H
