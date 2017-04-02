#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "definitions.h"

class Topology
{
public:

	virtual ~Topology() {}

	void init(IntVector2D size);

	IntVector2D getSize() const;

	void correctPosition(QVector3D& pos) const;
	IntVector2D correctPositionWithIntPrecision(QVector3D const& pos) const;
	IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const;
	void correctDisplacement(QVector3D& displacement) const;
	QVector3D displacement(QVector3D fromPoint, QVector3D toPoint) const;
    qreal distance(QVector3D fromPoint, QVector3D toPoint) const;
    QVector3D correctionIncrement (QVector3D pos1, QVector3D pos2) const;

	void serializePrimitives(QDataStream& stream) const;
	void deserializePrimitives(QDataStream& stream);

private:
	inline void correctPosition(IntVector2D & pos) const;

	IntVector2D _size { 0, 0 };
};

void Topology::correctPosition(IntVector2D & pos) const
{
	pos.x = ((pos.x % _size.x) + _size.x) % _size.x;
	pos.y = ((pos.y % _size.y) + _size.y) % _size.y;
}

#endif // TOPOLOGY_H
