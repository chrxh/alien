#include "topology.h"

#include "model/entities/cell.h"

Topology::Topology(IntVector2D size)
	: _size(size)
{
}


IntVector2D Topology::init(IntVector2D size)
{
	_size = size;
}

IntVector2D Topology::getSize() const
{
	return _size;
}

void Topology::correctPosition(QVector3D & pos) const
{
	IntVector2D intPart { qFloor(pos.x()), qFloor(pos.y()) };
	qreal fracPartX = pos.x() - intPart.x;
	qreal fracPartY = pos.y() - intPart.y;
	correctPosition(intPart);
	pos.setX(static_cast<qreal>(intPart.x) + fracPartX);
	pos.setY(static_cast<qreal>(intPart.y) + fracPartY);
}

IntVector2D Topology::correctPositionWithIntPrecision(QVector3D const& pos) const
{
	IntVector2D intPos{ qFloor(pos.x()), qFloor(pos.y()) };
	correctPosition(intPos);
	return intPos;
}

IntVector2D Topology::shiftPosition(IntVector2D const & pos, IntVector2D const && shift) const
{
	IntVector2D temp{ pos.x + shift.x, pos.y + shift.y };
	correctPosition(temp);
	return temp;
}

void Topology::correctDisplacement(QVector3D & displacement) const
{
	IntVector2D intDisplacement{ qFloor(displacement.x()), qFloor(displacement.y()) };
	qreal rx = displacement.x() - static_cast<qreal>(intDisplacement.x);
	qreal ry = displacement.y() - static_cast<qreal>(intDisplacement.y);
	intDisplacement.x += _size.x / 2;
	intDisplacement.y += _size.y / 2;
	correctPosition(intDisplacement);
	intDisplacement.x -= _size.x / 2;
	intDisplacement.y -= _size.y / 2;
	displacement.setX(static_cast<qreal>(intDisplacement.x) + rx);
	displacement.setY(static_cast<qreal>(intDisplacement.y) + ry);
}

QVector3D Topology::displacement(QVector3D fromPoint, QVector3D toPoint) const
{
	QVector3D d = toPoint - fromPoint;
	correctDisplacement(d);
	return d;
}

qreal Topology::distance(Vector3D fromPoint, QVector3D toPoint) const
{
    return displacement(fromPoint, toPoint).length();
}
/*
QVector3D Topology::displacement(Cell * fromCell, Cell * toCell) const
{
	return displacement(fromCell->calcPosition(), toCell->calcPosition());
}

qreal Topology::distance(Cell * fromCell, Cell * toCell) const
{
	return displacement(fromCell, toCell).length();
}
*/

QVector3D Topology::correctionIncrement (QVector3D pos1, QVector3D pos2) const
{
    QVector3D correction;
    if( (pos2.x()-pos1.x()) > (_size.x/2.0) )
        correction.setX(-_topology->getSize().x);
    if( (pos1.x()-pos2.x()) > (_size.x/2.0) )
        correction.setX(_topology->getSize().x);
    if( (pos2.y()-pos1.y()) > (_size.y/2.0) )
        correction.setY(-_topology->getSize().y);
    if( (pos1.y()-pos2.y()) > (_size.y/2.0) )
        correction.setY(_topology->getSize().y);
    return correction;
}

void Topology::serializePrimitives(QDataStream & stream) const
{
	stream << _size.x << _size.y;
}

void Topology::deserializePrimitives(QDataStream & stream)
{
	stream >> _size.x >> _size.y;
}


