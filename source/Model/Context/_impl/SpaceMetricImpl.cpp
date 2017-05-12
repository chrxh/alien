#include "SpaceMetricImpl.h"

#include "model/Entities/Cell.h"

SpaceMetricImpl::SpaceMetricImpl(QObject * parent)
	: SpaceMetric(parent)
{
}

void SpaceMetricImpl::init(IntVector2D size)
{
	_size = size;
}

SpaceMetric * SpaceMetricImpl::clone(QObject * parent) const
{
	auto metric = new SpaceMetricImpl(parent);
	metric->_size = _size;
	return metric;
}

IntVector2D SpaceMetricImpl::getSize() const
{
	return _size;
}

void SpaceMetricImpl::correctPosition(QVector2D & pos) const
{
	IntVector2D intPart{ qFloor(pos.x()), qFloor(pos.y()) };
	qreal fracPartX = pos.x() - intPart.x;
	qreal fracPartY = pos.y() - intPart.y;
	correctPosition(intPart);
	pos.setX(static_cast<qreal>(intPart.x) + fracPartX);
	pos.setY(static_cast<qreal>(intPart.y) + fracPartY);
}

IntVector2D SpaceMetricImpl::correctPositionWithIntPrecision(QVector2D const& pos) const
{
	IntVector2D intPos{ qFloor(pos.x()), qFloor(pos.y()) };
	correctPosition(intPos);
	return intPos;
}

IntVector2D SpaceMetricImpl::shiftPosition(IntVector2D const & pos, IntVector2D const && shift) const
{
	IntVector2D temp{ pos.x + shift.x, pos.y + shift.y };
	correctPosition(temp);
	return temp;
}

void SpaceMetricImpl::correctDisplacement(QVector2D & displacement) const
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

QVector2D SpaceMetricImpl::displacement(QVector2D fromPoint, QVector2D toPoint) const
{
	QVector2D d = toPoint - fromPoint;
	correctDisplacement(d);
	return d;
}

qreal SpaceMetricImpl::distance(QVector2D fromPoint, QVector2D toPoint) const
{
	return displacement(fromPoint, toPoint).length();
}

QVector2D SpaceMetricImpl::correctionIncrement(QVector2D pos1, QVector2D pos2) const
{
	QVector2D correction;
	if ((pos2.x() - pos1.x()) > (_size.x / 2.0))
		correction.setX(-_size.x);
	if ((pos1.x() - pos2.x()) > (_size.x / 2.0))
		correction.setX(_size.x);
	if ((pos2.y() - pos1.y()) > (_size.y / 2.0))
		correction.setY(-_size.y);
	if ((pos1.y() - pos2.y()) > (_size.y / 2.0))
		correction.setY(_size.y);
	return correction;
}

void SpaceMetricImpl::serializePrimitives(QDataStream & stream) const
{
	stream << _size.x << _size.y;
}

void SpaceMetricImpl::deserializePrimitives(QDataStream & stream)
{
	stream >> _size.x >> _size.y;
}


