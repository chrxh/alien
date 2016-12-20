#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "definitions.h"

class Topology
{
public:
	Topology(IntVector2D size);
	virtual ~Topology() {}

	IntVector2D getSize() const;

	void correctPosition(QVector3D& pos) const;
	IntVector2D correctPositionWithIntPrecision(QVector3D const& pos) const;
	IntVector2D shiftPosition(IntVector2D const& pos, IntVector2D const && shift) const;
	void correctDisplacement(QVector3D& displacement) const;
	QVector3D displacement(QVector3D fromPoint, QVector3D toPoint) const;
	QVector3D displacement(Cell* fromCell, Cell* toCell) const;
	qreal distance(Cell* fromCell, Cell* toCell) const;

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
